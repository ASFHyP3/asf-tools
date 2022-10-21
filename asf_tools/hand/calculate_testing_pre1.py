"""Calculate Height Above Nearest Drainage (HAND) from the Copernicus GLO-30 DEM"""
import argparse
import logging
import os
import sys
import urllib
import warnings
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Union

import astropy.convolution
import fiona
import numpy as np
import rasterio.crs
import rasterio.mask
from pysheds.sgrid import sGrid as Sgrid
from shapely.geometry import GeometryCollection, shape

from asf_tools.composite import write_cog
from asf_tools.dem import prepare_dem_vrt

log = logging.getLogger(__name__)


def fill_nan(array: np.ndarray) -> np.ndarray:
    """Replace NaNs with values interpolated from their neighbors

    Replace NaNs with values interpolated from their neighbors using a 2D Gaussian
    kernel, see: https://docs.astropy.org/en/stable/convolution/#using-astropy-s-convolution-to-replace-bad-data
    """
    kernel = astropy.convolution.Gaussian2DKernel(x_stddev=3)  # kernel x_size=8*stddev
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        array = astropy.convolution.interpolate_replace_nans(
            array, kernel, convolve=astropy.convolution.convolve
        )

    return array


def fill_nan_based_on_dem(arr, dem):
    """
    filled_arr=fill_nan_based_on_DEM(arr, dem)
    Fills Not-a-number values in arr using astropy.
    """
    hond = dem - arr
    kernel = astropy.convolution.Gaussian2DKernel(x_stddev=3)
    arr_type = hond.dtype
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while np.any(np.isnan(hond)):
            hond = astropy.convolution.interpolate_replace_nans(hond.astype(float),
                                                                kernel, convolve=astropy.convolution.convolve)
    my_mask = np.isnan(arr)
    ch = np.logical_and(my_mask, dem - hond < 0)
    indices = np.where(ch)
    if np.any(ch):
        hond[indices] = dem[indices]

    arr[my_mask] = dem[my_mask]-hond[my_mask]

    return arr.astype(arr_type)


def fiona_read_vectorfile(vectorfile, get_property=None):
    """shapes=fiona_read_vectorfile(vectorfile, get_property=None)
       shapes, props=fiona_read_vectorfile(vectorfile, get_property='Property_Name')
       Returns a list of shapes (and optionally properties) using fiona.

       vectorfile: any fiona compatible vector file.
       get_property: String for the property to be read.
       shapes: List of vector "geometry"
       props:  List of vector "properties"
    """
    with fiona.open(vectorfile, "r") as shpf:
        shapes = [feature["geometry"] for feature in shpf]
        if get_property is not None:
            props = [feature["properties"][get_property] for feature in shpf]
            return shapes, props
        else:
            return shapes


def calculate_hand(demfile, mask=None, nodata=None, acc_thresh: Optional[int] = 100):

    grid = Sgrid.from_raster(demfile)

    # if mask is not None:
    #    grid.mask = mask

    if nodata:
        grid.nodata = nodata

    dem = grid.read_raster(demfile)

    log.info('Fill pits in DEM')
    pit_filled_dem = grid.fill_pits(dem)

    log.info('Filling depressions')
    flooded_dem = grid.fill_depressions(pit_filled_dem)

    # free useless memory
    pit_filled_dem = None

    log.info('Resolving flats')
    inflated_dem = grid.resolve_flats(flooded_dem)

    # free memory
    flooded_dem = None

    log.info('Obtaining flow direction')
    flow_dir = grid.flowdir(inflated_dem, apply_mask=True)

    log.info('Calculating flow accumulation')
    acc = grid.accumulation(flow_dir)

    if acc_thresh is None:
        acc_thresh = acc.mean()

    log.info(f'Calculating HAND using accumulation threshold of {acc_thresh}')
    hand = grid.compute_hand(flow_dir, inflated_dem, acc > acc_thresh, inplace=False)


    # fill rivers
    if np.isnan(hand).any():
        # get nans inside masked area and find mean height for pixels outside the nans (but inside basin mask)
        valid_nanmask = np.logical_and(mask, np.isnan(hand))
        valid_mask = np.logical_and(mask, ~np.isnan(hand))
        mean_height = inflated_dem[valid_mask].mean()

        # calculate gradient and set mean gradient magnitude as threshold for flatness.
        g0, g1 = np.gradient(inflated_dem)
        gMag = np.sqrt(g0 ** 2 + g1 ** 2)
        # gMagTh = np.min([1, np.mean(gMag * np.isnan(hand))])
        gMag[~np.isnan(hand)] = np.nan
        gMagTh = np.min([1, np.nanmean(gMag)])
        valid_flats = np.logical_and(valid_nanmask, gMag < gMagTh)
        valid_low_flats = np.logical_and(valid_flats, inflated_dem < mean_height)
        hand[valid_low_flats] = 0

    return hand, acc


def get_hand_by_land_mask(hand, nodata_fill_value, dem):
    # Download GSHHG
    gshhg_dir = '/media/jzhu4/data/hand/external_data'
    # gshhg_url = 'http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip'
    gshhg_url = 'https://www.ngdc.noaa.gov/mgg/shorelines/data/gshhg/latest/gshhg-shp-2.3.7.zip'
    gshhg_zipfile = os.path.join(gshhg_dir, "gshhg-shp-2.3.7.zip")
    gshhg_file = os.path.join(gshhg_dir, "GSHHS_shp/f/GSHHS_f_L1.shp")

    if not os.path.exists(gshhg_file):
        if not os.path.exists(gshhg_zipfile):
            os.system(f'mkdir -p {gshhg_dir}')
            urllib.request.urlretrieve(gshhg_url, gshhg_zipfile)

        with zipfile.ZipFile(gshhg_zipfile, 'r') as zip_ref:
            zip_ref.extractall(path=gshhg_dir)

    gshhg = fiona_read_vectorfile(gshhg_file)
    # generate land_mask for the DEM. invert=If False (default), mask will be False inside shapes and True outside
    land_mask, tf, win = rasterio.mask.raster_geometry_mask(dem, gshhg, crop=False,
                                                            invert=True)
    # set ocean/sea values in hand to epsilon, sea_mask=np.invert(land_mask)
    hand[np.invert(land_mask)] = nodata_fill_value

    return hand


def get_basin_dem_file(dem, basin_affine_tf, basin_array, basin_dem_file):
    """write the basin_dem geotiff
    """
    out_meta = dem.meta.copy()
    nodata_value = out_meta['nodata']
    if not nodata_value:
        nodata_value = np.finfo(float).eps
    out_meta.update({
        'driver': 'GTiff',
        'width': basin_array.shape[1],
        'height': basin_array.shape[0],
        'transform': basin_affine_tf,
        'nodata': nodata_value
    })
    with rasterio.open(fp=basin_dem_file, mode='w', **out_meta) as dst:
        dst.write(basin_array, 1)

    return basin_dem_file


def calculate_hand_for_basins(out_raster:  Union[str, Path], geometries: GeometryCollection,
                              dem_file: Union[str, Path]):
    """Calculate the Height Above Nearest Drainage (HAND) for watershed boundaries (hydrobasins).

    For watershed boundaries, see: https://www.hydrosheds.org/page/hydrobasins

    Args:
        out_raster: HAND GeoTIFF to create
        geometries: watershed boundary (hydrobasin) polygons to calculate HAND over
        dem_file: DEM raster covering (containing) `geometries`
    """
    with rasterio.open(dem_file) as src:
        basin_mask, basin_affine_tf, basin_window = rasterio.mask.raster_geometry_mask(
            src, geometries, all_touched=True, crop=True, pad=True, pad_width=1
        )

        basin_array = src.read(1, window=basin_window)

        # produce tmp_dem.tif based on basin_array and basin_affine_tf
        basin_dem_file = "/tmp/tmp_dem.tif"
        get_basin_dem_file(src, basin_affine_tf, basin_array, basin_dem_file)

        hand, acc = calculate_hand(basin_dem_file, ~basin_mask)

        # write the HAND before fill_nan_based_dem
        dirn = os.path.dirname(out_raster)
        prefix = os.path.basename(out_raster).split(".tif")[0]
        out_by_write_cog = os.path.join(dirn, f"{prefix}_hand_before_fill.tif")
        write_cog(out_by_write_cog, hand, transform=basin_affine_tf.to_gdal(), epsg_code=src.crs.to_epsg())

        # write the acc
        dirn = os.path.dirname(out_raster)
        prefix = os.path.basename(out_raster).split(".tif")[0]
        acc_out_by_write_cog = os.path.join(dirn, f"{prefix}_acc.tif")
        write_cog(acc_out_by_write_cog, acc, transform=basin_affine_tf.to_gdal(), epsg_code=src.crs.to_epsg())

        # fill non basin_mask with nodata_fill_value
        nodata_fill_value = np.finfo(float).eps
        hand[basin_mask] = nodata_fill_value

        if np.isnan(hand).any():
            basin_dem = rasterio.open(basin_dem_file, 'r')

            # fill ocean pixels with the minimum value of data type of float32
            # hand = get_hand_by_land_mask(hand, nodata_fill_value, basin_dem)

            # fill nan pixels
            hand = fill_nan_based_on_dem(hand, basin_dem.read(1))

        # fill basin_mask with nan
        hand[basin_mask] = np.nan
        # write the HAND
        write_cog(str(out_raster), hand, transform=basin_dem.get_transform(), epsg_code=basin_dem.crs.to_epsg())


def make_copernicus_hand(out_raster:  Union[str, Path], vector_file: Union[str, Path]):
    """Copernicus GLO-30 Height Above Nearest Drainage (HAND)

    Make a Height Above Nearest Drainage (HAND) GeoTIFF from the Copernicus GLO-30 DEM
    covering the watershed boundaries (hydrobasins) defined in a vector file.

    For watershed boundaries, see: https://www.hydrosheds.org/page/hydrobasins

    Args:
        out_raster: HAND GeoTIFF to create
        vector_file: Vector file of watershed boundary (hydrobasin) polygons to calculate HAND over
    """
    with fiona.open(vector_file) as vds:
        geometries = GeometryCollection([shape(feature['geometry']) for feature in vds])

    with NamedTemporaryFile(suffix='.vrt', delete=False) as dem_vrt:
        prepare_dem_vrt(dem_vrt.name, geometries)
        calculate_hand_for_basins(out_raster, geometries, dem_vrt.name)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='For watershed boundaries, see: https://www.hydrosheds.org/page/hydrobasins',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('out_raster', help='HAND GeoTIFF to create')
    parser.add_argument('vector_file', help='Vector file of watershed boundary (hydrobasin) polygons to calculate HAND '
                                            'over. Vector file Must be openable by GDAL, see: '
                                            'https://gdal.org/drivers/vector/index.html')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    log.debug(' '.join(sys.argv))
    log.info(f'Calculating HAND for {args.vector_file}')

    make_copernicus_hand(args.out_raster, args.vector_file)

    log.info(f'HAND GeoTIFF created successfully: {args.out_raster}')
