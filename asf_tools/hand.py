"""Calculate height above nearest drainage (HAND) from the Copernicus GLO-30 Public DEM"""
import argparse
import logging
import sys
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Union

import astropy.convolution
import fiona
import numpy as np
import rasterio.crs
import rasterio.mask
from pysheds.grid import Grid
from shapely.geometry import GeometryCollection, shape

from asf_tools.composite import write_cog
from asf_tools.dem import prepare_dem_vrt

log = logging.getLogger(__name__)


def fill_nan(arr):
    """
    filled_arr=fill_nan(arr)
    Fills Not-a-number values in arr using astropy.
    """

    kernel = astropy.convolution.Gaussian2DKernel(x_stddev=3)  # kernel x_size=8*stddev
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        arr = astropy.convolution.interpolate_replace_nans(
            arr, kernel, convolve=astropy.convolution.convolve
        )

    return arr


def calculate_hand(dem_array, dem_affine: rasterio.Affine, dem_crs: rasterio.crs.CRS, basin_mask, acc_thresh=100):
    """
    hand=calculate_hand(dem_array, dem_gT, dem_proj4, mask=None, verbose=False)
    Calculate the height above nearest drainage using pySHEDS library. This is done over a few steps:

    Fill_Depressions fills depressions in a DEM (regions of cells lower than their surrounding neighbors).
    Resolve_Flats resolves drainable flats in a DEM.
    FlowDir converts the DEM to flow direction based on dirmap.
    Accumulation converts from flow direction to flow accumulation.
    Compute_Hand is used to convert directions to height above nearest drainage.

    NaN values are filled at the end of resolve_flats and final steps.

    Inputs:
      dem_array=Numpy array of Digital Elevation Model (DEM) to convert to HAND.
      dem_gT= GeoTransform of the input DEM
      dem_proj4=Proj4 string of DEM
      mask=If provided parts of DEM can be masked out. If not entire DEM is evaluated.
      verbose=If True, provides information about where NaN values are encountered.
      acc_thresh=Accumulation threshold. By default is set to 100. If none,
                 mean value of accumulation array (acc.mean()) is used.
    """

    grid = Grid()
    grid.add_gridded_data(dem_array, data_name='dem', affine=dem_affine, crs=dem_crs.to_dict(), mask=~basin_mask)

    grid.fill_depressions('dem', out_name='flooded_dem')
    if np.isnan(grid.flooded_dem).any():
        log.debug('NaNs encountered in flooded DEM; filling.')
        grid.flooded_dem = fill_nan(grid.flooded_dem)

    # # https://github.com/mdbartos/pysheds/issues/118
    # try:
    #     grid.resolve_flats('flooded_dem', out_name='inflated_dem')
    # except:
    #     grid.inflated_dem = grid.flooded_dem
    log.info('Resolving flats')
    grid.resolve_flats('flooded_dem', out_name='inflated_dem')
    if np.isnan(grid.inflated_dem).any():
        log.debug('NaNs encountered in inflated DEM; replacing NaNs with original DEM values')
        grid.inflated_dem[np.isnan(grid.inflated_dem)] = dem_array[np.isnan(grid.inflated_dem)]

    log.info('Obtaining flow direction')
    grid.flowdir(data='inflated_dem', out_name='dir', apply_mask=True)
    if np.isnan(grid.dir).any():
        log.debug('NaNs encountered in flow direction; filling.')
        grid.dir = fill_nan(grid.dir)

    log.info('Calculating flow accumulation')
    grid.accumulation(data='dir', out_name='acc')
    if np.isnan(grid.acc).any():
        log.debug('NaNs encountered in accumulation; filling.')
        grid.acc = fill_nan(grid.acc)

    if acc_thresh is None:
        acc_thresh = grid.acc.mean()

    log.info(f'Calculating HAND using accumulation threshold of {acc_thresh}')
    hand = grid.compute_hand('dir', 'inflated_dem', grid.acc > acc_thresh, inplace=False)
    if np.isnan(hand).any():
        log.debug('NaNs encountered in HAND; filling.')
        hand = fill_nan(hand)

    # ensure non-basin is masked after fill_nan
    hand[basin_mask] = np.nan

    return hand


def calculate_hand_for_basins(out_raster:  Union[str, Path], geometries: GeometryCollection,
                              dem_file: Union[str, Path]):
    """Calculate HAND data for drainage basins"""

    with rasterio.open(dem_file) as src:
        basin_mask, basin_affine_tf, dem_window = rasterio.mask.raster_geometry_mask(
            src, geometries, all_touched=True, crop=True, pad=True, pad_width=1
        )
        dem_array = src.read(1, window=dem_window)
        dem_affine_tf = src.transform
        dem_crs = src.crs

    hand = calculate_hand(dem_array, dem_affine_tf, dem_crs, basin_mask)

    write_cog(str(out_raster), hand, transform=basin_affine_tf.to_gdal(), epsg_code=dem_crs.to_epsg())


def make_hand(out_raster:  Union[str, Path], vector_file: Union[str, Path]):
    with fiona.open(vector_file) as vds:
        geometries = GeometryCollection([shape(feature['geometry']) for feature in vds])

    with NamedTemporaryFile(suffix='.vrt', delete=False) as f:
        dem_file = Path(f.name)
        dem_file.touch()

    prepare_dem_vrt(dem_file, geometries)

    calculate_hand_for_basins(out_raster, geometries, dem_file)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('out_raster', type=Path,
                        help='HAND GeoTIFF to create')
    parser.add_argument('vector_file', type=Path,
                        help='Vector file of watershed boundary (hydrobasin) polygons to calculate HAND over.'
                             'Vector file Must be openable by GDAL, see: https://gdal.org/drivers/vector/index.html')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    log.debug(' '.join(sys.argv))
    log.info(f'Calculating HAND for {args.vector_file}')

    make_hand(args.out_raster, args.vector_file)

    log.info(f'HAND GeoTIFF created successfully: {args.out_raster}')
