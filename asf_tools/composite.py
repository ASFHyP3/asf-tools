"""Create S1 SAR Composite Mosaic using inverse area weighting ala David Small.

   Path vs infiles:
     If path is passed, code assumes files are in an ASF HyP3 RTC Stacking arrangement.
     i.e  {path}/20*/PRODUCT/ contains the input RTC data and the area maps or
          {path}/S1?_IW_*RTC*/ contains the input RTC data and the area maps


"""

import argparse
import glob
import logging
import os
import statistics
from tempfile import TemporaryDirectory
from typing import List

import numpy as np
from osgeo import gdal, osr


def get_epsg_code(info: dict) -> int:
    """Get the EPSG code from a GDAL Info dictionary"""
    proj = osr.SpatialReference(info['coordinateSystem']['wkt'])
    epsg_code = int(proj.GetAttrValue('AUTHORITY', 1))
    return epsg_code


def get_target_epsg_code(codes: List[int]) -> int:
    """Determine the target UTM EPSG projection for the output mosaic

    Args:
        codes: List of UTM EPSG codes

    Returns:
        target: UTM EPSG code
    """
    # use median east/west UTM zone of all files, regardless of hemisphere
    # UTM EPSG codes for each hemisphere will look like:
    #   North: 326XX
    #   South: 327XX
    code_array = np.array(codes)
    valid_codes = np.concatenate([np.arange(32601, 32661), np.arange(32701, 32761)])
    if not np.isin(code_array, valid_codes).all():
        raise ValueError(f'Non UTM EPSG code encountered: {codes}')
    hemispheres = code_array // 100 * 100
    zones = code_array % 100

    # handle antimeridian
    target_zone = int(statistics.median(zones % 60))
    if target_zone == 0:
        target_zone = 60

    target_hemisphere = int(statistics.mode(hemispheres))

    return target_hemisphere + target_zone


def get_area_raster(raster: str) -> str:
    '''Determine the path of the area raster for a given backscatter raster based on naming conventions for HyP3 RTC
    products

    Args:
        raster: path of the backscatter raster, e.g. S1A_IW_20181102T155531_DVP_RTC30_G_gpuned_5685_VV.tif

    Returns:
        area_raster: path of the area raster, e.g. S1A_IW_20181102T155531_DVP_RTC30_G_gpuned_5685_area.tif
    '''
    return '_'.join(raster.split('_')[:-1] + ['area.tif'])


def get_full_extent(raster_info: dict):
    upper_left_corners = [info['cornerCoordinates']['upperLeft'] for info in raster_info.values()]
    lower_right_corners = [info['cornerCoordinates']['lowerRight'] for info in raster_info.values()]

    ulx = min([ul[0] for ul in upper_left_corners])
    uly = max([ul[1] for ul in upper_left_corners])
    lrx = max([lr[0] for lr in lower_right_corners])
    lry = min([lr[1] for lr in lower_right_corners])

    logging.debug(f"Full extent raster upper left: ({ulx, uly}); lower right: ({lrx, lry})")

    trans = []
    proj = ''
    for info in raster_info.values():
        # Only need info from any one raster
        trans = info['geoTransform']
        proj = info['coordinateSystem']['wkt']
        break

    trans[0] = ulx
    trans[3] = uly

    return (ulx, uly), (lrx, lry), trans, proj


def reproject_to_target(raster_info: dict, target_epsg_code: int, target_resolution: float, directory: str) -> dict:
    logging.info("Checking projections")
    target_raster_info = {}
    for raster, info in raster_info.items():
        epsg_code = get_epsg_code(info)
        resolution = info['geoTransform'][1]
        if epsg_code != target_epsg_code or resolution != target_resolution:
            logging.info(f"Reprojecting {raster}")
            reprojected_raster = os.path.join(directory, os.path.basename(raster))
            gdal.Warp(
                reprojected_raster, raster, dstSRS=f'EPSG:{target_epsg_code}',
                xRes=target_resolution, yRes=target_resolution, targetAlignedPixels=True
            )

            area_raster = get_area_raster(raster)
            logging.info(f"Reprojecting {area_raster}")
            reprojected_area_raster = os.path.join(directory, os.path.basename(area_raster))
            gdal.Warp(
                reprojected_area_raster, area_raster, dstSRS=f'EPSG:{target_epsg_code}',
                xRes=target_resolution, yRes=target_resolution, targetAlignedPixels=True
            )

            target_raster_info[reprojected_raster] = gdal.Info(reprojected_raster,  format='json')
        else:
            logging.info(f"No need to reproject {raster}")
            target_raster_info[raster] = info

    return target_raster_info


def write_cog(outfile: str, data: np.ndarray, transform: List[float], projection: str,
              dtype=gdal.GDT_Float32, nodata_value=None):
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(
        outfile, data.shape[1], data.shape[0], 1, dtype,
        options=["TILED=YES", "COMPRESS=LZW", "INTERLEAVE=BAND"]
    )
    out_raster.GetRasterBand(1).WriteArray(data)
    if nodata_value is not None:
        out_raster.GetRasterBand(1).SetNoDataValue(nodata_value)
    out_raster.SetGeoTransform(transform)
    out_raster.SetProjection(projection)
    del out_raster  # How to close w/ gdal


def make_composite(outfile, rasters=None, resolution=None):

    '''Create a composite mosaic of rasters using inverse area weighting to adjust backscatter'''

    logging.info(f"make_composite: {outfile} {rasters} {resolution}")

    raster_info = {}
    for raster in rasters:
        raster_info[raster] = gdal.Info(raster, format='json')

    target_epsg_code = get_target_epsg_code([get_epsg_code(info) for info in raster_info.values()])
    if resolution is None:
        resolution = max([info['geoTransform'][1] for info in raster_info.values()])

    # resample rasters to maximum resolution & common UTM zone
    with TemporaryDirectory(prefix='reprojected_') as temp_dir:
        raster_info = reproject_to_target(raster_info, target_epsg_code=target_epsg_code, target_resolution=resolution,
                                          directory=temp_dir)

        # Get extent of union of all images
        full_ul, full_lr, full_trans, full_proj = get_full_extent(raster_info)

        nx = int(abs(full_ul[0] - full_lr[0]) // resolution)
        ny = int(abs(full_ul[1] - full_lr[1]) // resolution)

        outputs = np.zeros((ny, nx))
        weights = np.zeros(outputs.shape)
        counts = np.zeros(outputs.shape, dtype=np.int8)

        logging.info("Calculating output values")
        for raster, info in raster_info.items():
            logging.info(f"Processing raster {raster}")
            logging.info(f"Raster upper left: {info['cornerCoordinates']['upperLeft']}; "
                         f"lower right: {info['cornerCoordinates']['lowerRight']}")

            logging.info(f"Reading raster values {raster}")
            rds = gdal.Open(raster)
            values = rds.GetRasterBand(1).ReadAsArray()
            del rds  # How to close w/ gdal

            area_raster = get_area_raster(raster)

            logging.info(f"Reading area raster {area_raster}")
            ads = gdal.Open(area_raster)
            areas = ads.GetRasterBand(1).ReadAsArray()
            del ads  # How to close w/ gdal

            ulx, uly = info['cornerCoordinates']['upperLeft']
            y_index_start = int((full_ul[1] - uly) // resolution)
            y_index_end = y_index_start + values.shape[0]

            x_index_start = int((ulx - full_ul[0]) // resolution)
            x_index_end = x_index_start + values.shape[1]

            logging.info(
                f"Placing values in output grid at {y_index_start}:{y_index_end} and {x_index_start}:{x_index_end}"
            )

            temp = 1.0/areas
            temp[values == 0] = 0
            mask = np.ones(values.shape, dtype=np.uint8)
            mask[values == 0] = 0

            outputs[y_index_start:y_index_end, x_index_start:x_index_end] += values * temp
            weights[y_index_start:y_index_end, x_index_start:x_index_end] += temp
            counts[y_index_start:y_index_end, x_index_start:x_index_end] += mask

    # Divide by the total weight applied
    outputs /= weights

    logging.info("Writing output files")
    write_cog(outfile, outputs, full_trans, full_proj, nodata_value=0)
    write_cog(outfile.replace('.tif', '_counts.tif'), counts, full_trans, full_proj, dtype=gdal.GDT_Int16)

    logging.info("Program successfully completed")


def get_rasters_from_path(path, pol):
    # Establish input file list
    rasters = glob.glob(os.path.join(path, f"S1?_IW_*RTC*/*{pol}.tif"))
    rasters.append(glob.glob(os.path.join(path, f"20*/PRODUCT/*{pol}.tif")))

    return rasters


def main():
    parser = argparse.ArgumentParser(
        prog="make_composite.py",
        description="Create a weighted composite mosaic from a set of S-1 RTC products",
        epilog="Output pixel values calculated using weights that are the inverse of the area."
    )

    parser.add_argument("outfile", help="Name of output weighted mosaic geotiff file")
    parser.add_argument("--pol", choices=['VV', 'VH', 'HH', 'HV'], default='VV',
                        help="When using multi-pol data, only mosaic given polarization")
    parser.add_argument("-r", "--resolution", type=float, help="Desired output resolution")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", "--path", help="Name of directory where input stack is located")
    group.add_argument("-i", "--infiles", nargs='*', help="Names of input series files")

    args = parser.parse_args()

    logFile = "make_composite_{}.log".format(os.getpid())
    logging.basicConfig(filename=logFile, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Starting run")

    rasters = get_rasters_from_path(args.path, args.pol) if args.path else args.infiles

    make_composite(args.outfile, rasters, args.path)
