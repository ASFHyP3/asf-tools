"""Create a local-resolution-weighted composite from Sentinel-1 RTC products.

Create a local-resolution-weighted composite from a set of Sentinel-1 RTC
products (D. Small, 2012). The local resolution, defined as the inverse of the
local contributing (scattering) area, is used to weight each RTC products'
contributions to the composite image on a pixel-by-pixel basis. The composite image
is created as a Cloud Optimized GeoTIFF (COG). Additionally, a COG specifying
the number of rasters contributing to each composite pixel is created.

References:
    David Small, 2012: <https://doi.org/10.1109/IGARSS.2012.6350465>
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from statistics import multimode
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import List, Union

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()
log = logging.getLogger(__name__)


def get_epsg_code(info: dict) -> int:
    """Get the EPSG code from a GDAL Info dictionary

    Args:
        info: The dictionary returned by a gdal.Info call

    Returns:
        epsg_code: The integer EPSG code
    """
    proj = osr.SpatialReference(info['coordinateSystem']['wkt'])
    epsg_code = int(proj.GetAttrValue('AUTHORITY', 1))
    return epsg_code


def epsg_to_wkt(epsg_code: int) -> str:
    """Get the WKT representation of a projection from its EPSG code

    Args:
        epsg_code: The integer EPSG code

    Returns:
        wkt: The WKT representation of the projection
    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    return srs.ExportToWkt()


def get_target_epsg_code(codes: List[int]) -> int:
    """Determine the target UTM EPSG projection for the output composite

    Args:
        codes: List of UTM EPSG codes

    Returns:
        target: UTM EPSG code
    """
    # use median east/west UTM zone of all files, regardless of hemisphere
    # UTM EPSG codes for each hemisphere will look like:
    #   North: 326XX
    #   South: 327XX
    valid_codes = list(range(32601, 32661)) + list(range(32701, 32761))
    if bad_codes := set(codes) - set(valid_codes):
        raise ValueError(f'Non UTM EPSG code encountered: {bad_codes}')

    hemispheres = [c // 100 * 100 for c in codes]
    # if even modes, choose lowest (North)
    target_hemisphere = min(multimode(hemispheres))

    zones = sorted([c % 100 for c in codes])
    # if even length, choose fist of median two
    target_zone = zones[(len(zones) - 1) // 2]

    return target_hemisphere + target_zone


def get_area_raster(raster: str) -> str:
    """Determine the path of the area raster for a given backscatter raster based on naming conventions for HyP3 RTC
    products

    Args:
        raster: path of the backscatter raster, e.g. S1A_IW_20181102T155531_DVP_RTC30_G_gpuned_5685_VV.tif

    Returns:
        area_raster: path of the area raster, e.g. S1A_IW_20181102T155531_DVP_RTC30_G_gpuned_5685_area.tif
    """
    return '_'.join(raster.split('_')[:-1] + ['area.tif'])


def get_full_extent(raster_info: dict):
    """Determine the corner coordinates and geotransform for the full extent of a set of rasters

    Args:
        raster_info: A dictionary of gdal.Info results for the set of rasters

    Returns:
        upper_left: The upper left corner of the extent as a tuple
        upper_right: The lower right corner of the extent as a tuple
        geotransform: The geotransform of the extent as a list
    """
    upper_left_corners = [info['cornerCoordinates']['upperLeft'] for info in raster_info.values()]
    lower_right_corners = [info['cornerCoordinates']['lowerRight'] for info in raster_info.values()]

    ulx = min([ul[0] for ul in upper_left_corners])
    uly = max([ul[1] for ul in upper_left_corners])
    lrx = max([lr[0] for lr in lower_right_corners])
    lry = min([lr[1] for lr in lower_right_corners])

    log.debug(f'Full extent raster upper left: ({ulx, uly}); lower right: ({lrx, lry})')

    trans = []
    for info in raster_info.values():
        # Only need info from any one raster
        trans = info['geoTransform']
        break

    trans[0] = ulx
    trans[3] = uly

    return (ulx, uly), (lrx, lry), trans


def reproject_to_target(raster_info: dict, target_epsg_code: int, target_resolution: float, directory: str) -> dict:
    """Reprojects a set of raster images to a common projection and resolution

    Args:
        raster_info: A dictionary of gdal.Info results for the set of rasters
        target_epsg_code: The integer EPSG code for the target projection
        target_resolution: The target resolution
        directory: The directory in which to create the reprojected files

    Returns:
        target_raster_info: An updated dictionary of gdal.Info results for the reprojected files
    """
    target_raster_info = {}
    for raster, info in raster_info.items():
        epsg_code = get_epsg_code(info)
        resolution = info['geoTransform'][1]
        if epsg_code != target_epsg_code or resolution != target_resolution:
            log.info(f'Reprojecting {raster}')
            reprojected_raster = os.path.join(directory, os.path.basename(raster))
            gdal.Warp(
                reprojected_raster, raster, dstSRS=f'EPSG:{target_epsg_code}',
                xRes=target_resolution, yRes=target_resolution, targetAlignedPixels=True
            )

            area_raster = get_area_raster(raster)
            log.info(f'Reprojecting {area_raster}')
            reprojected_area_raster = os.path.join(directory, os.path.basename(area_raster))
            gdal.Warp(
                reprojected_area_raster, area_raster, dstSRS=f'EPSG:{target_epsg_code}',
                xRes=target_resolution, yRes=target_resolution, targetAlignedPixels=True
            )

            target_raster_info[reprojected_raster] = gdal.Info(reprojected_raster,  format='json')
        else:
            log.info(f'No need to reproject {raster}')
            target_raster_info[raster] = info

    return target_raster_info


def read_as_array(raster: str, band: int = 1) -> np.array:
    """Reads data from a raster image into memory

    Args:
        raster: The file path to a raster image
        band: The raster band to read

    Returns:
        data: The raster pixel data as a numpy array
    """
    log.debug(f'Reading raster values from {raster}')
    ds = gdal.Open(raster)
    data = ds.GetRasterBand(band).ReadAsArray()
    del ds  # How to close w/ gdal
    return data


def write_cog(file_name: Union[str, Path], data: np.ndarray, transform: List[float], epsg_code: int,
              dtype=gdal.GDT_Float32, nodata_value=None):
    """Creates a Cloud Optimized GeoTIFF

    Args:
        file_name: The output file name
        data: The raster data
        transform: The geotransform for the output GeoTIFF
        epsg_code: The integer EPSG code for the output GeoTIFF projection
        dtype: The pixel data type for the output GeoTIFF
        nodata_value: The NODATA value for the output Geotiff

    Returns:
        file_name: The output file name
    """
    log.info(f'Creating {file_name}')

    with NamedTemporaryFile() as temp_file:
        driver = gdal.GetDriverByName('GTiff')
        temp_geotiff = driver.Create(temp_file.name, data.shape[1], data.shape[0], 1, dtype)
        temp_geotiff.GetRasterBand(1).WriteArray(data)
        if nodata_value is not None:
            temp_geotiff.GetRasterBand(1).SetNoDataValue(nodata_value)
        temp_geotiff.SetGeoTransform(transform)
        temp_geotiff.SetProjection(epsg_to_wkt(epsg_code))

        driver = gdal.GetDriverByName('COG')
        options = ['COMPRESS=LZW', 'OVERVIEW_RESAMPLING=AVERAGE', 'NUM_THREADS=ALL_CPUS', 'BIGTIFF=YES']
        driver.CreateCopy(str(file_name), temp_geotiff, options=options)

        del temp_geotiff  # How to close w/ gdal
    return file_name


def make_composite(out_name: str, rasters: List[str], resolution: float = None):
    """Creates a local-resolution-weighted composite from Sentinel-1 RTC products

    Args:
        out_name: The base name of the output GeoTIFFs
        rasters: A list of file paths of the images to composite
        resolution: The pixel size for the output GeoTIFFs

    Returns:
        out_raster: Path to the created composite backscatter GeoTIFF
        out_counts_raster: Path to the created GeoTIFF with counts of scenes contributing to each pixel
    """
    if not rasters:
        raise ValueError('Must specify at least one raster to composite')

    raster_info = {}
    for raster in rasters:
        raster_info[raster] = gdal.Info(raster, format='json')
        # make sure gdal can read the area raster
        gdal.Info(get_area_raster(raster))

    target_epsg_code = get_target_epsg_code([get_epsg_code(info) for info in raster_info.values()])
    log.debug(f'Composite projection is EPSG:{target_epsg_code}')

    if resolution is None:
        resolution = max([info['geoTransform'][1] for info in raster_info.values()])
    log.debug(f'Composite resolution is {resolution} meters')

    # resample rasters to maximum resolution & common UTM zone
    with TemporaryDirectory(prefix='reprojected_') as temp_dir:
        raster_info = reproject_to_target(raster_info, target_epsg_code=target_epsg_code, target_resolution=resolution,
                                          directory=temp_dir)

        # Get extent of union of all images
        full_ul, full_lr, full_trans = get_full_extent(raster_info)

        nx = int(abs(full_ul[0] - full_lr[0]) // resolution)
        ny = int(abs(full_ul[1] - full_lr[1]) // resolution)

        outputs = np.zeros((ny, nx))
        weights = np.zeros(outputs.shape)
        counts = np.zeros(outputs.shape, dtype=np.int8)

        for raster, info in raster_info.items():
            log.info(f'Processing raster {raster}')
            log.debug(f"Raster upper left: {info['cornerCoordinates']['upperLeft']}; "
                      f"lower right: {info['cornerCoordinates']['lowerRight']}")

            values = read_as_array(raster)

            area_raster = get_area_raster(raster)
            areas = read_as_array(area_raster)

            ulx, uly = info['cornerCoordinates']['upperLeft']
            y_index_start = int((full_ul[1] - uly) // resolution)
            y_index_end = y_index_start + values.shape[0]

            x_index_start = int((ulx - full_ul[0]) // resolution)
            x_index_end = x_index_start + values.shape[1]

            log.debug(
                f'Placing values in output grid at {y_index_start}:{y_index_end} and {x_index_start}:{x_index_end}'
            )

            mask = values == 0
            raster_weights = 1.0 / areas
            raster_weights[mask] = 0

            outputs[y_index_start:y_index_end, x_index_start:x_index_end] += values * raster_weights
            weights[y_index_start:y_index_end, x_index_start:x_index_end] += raster_weights
            counts[y_index_start:y_index_end, x_index_start:x_index_end] += ~mask

    del values, areas, mask, raster_weights

    # Divide by the total weight applied
    outputs /= weights
    del weights

    out_raster = write_cog(f'{out_name}.tif', outputs, full_trans, target_epsg_code, nodata_value=0)
    del outputs

    out_counts_raster = write_cog(f'{out_name}_counts.tif', counts, full_trans, target_epsg_code, dtype=gdal.GDT_Int16)
    del counts

    return out_raster, out_counts_raster


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('out_name', help='Base name of output composite GeoTIFF (without extension)')
    parser.add_argument('rasters', nargs='+', help='Sentinel-1 GeoTIFF rasters to composite')
    parser.add_argument('-r', '--resolution', type=float,
                        help='Desired output resolution in meters '
                             '(default is the max resolution of all the input files)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    log.debug(' '.join(sys.argv))
    log.info(f'Creating a composite of {len(args.rasters)} rasters')

    raster, counts = make_composite(args.out_name, args.rasters, args.resolution)

    log.info(f'Composite created successfully: {raster}')
    log.info(f'Number of rasters contributing to each pixel: {counts}')
