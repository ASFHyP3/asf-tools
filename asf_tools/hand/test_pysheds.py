
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
from pysheds.pgrid import Grid as Pgrid

from scipy import ndimage
from shapely.geometry import GeometryCollection, shape

from asf_tools.composite import write_cog
from asf_tools.dem import prepare_dem_vrt

log = logging.getLogger(__name__)


def get_basin_dem_file(dem, basin_affine_tf, basin_array, basin_dem_file):
    # optional, fill_nan
    # produce tmp_dem.tif based on basin_array and basin_affine_tf
    out_meta = dem.meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'width': basin_array.shape[1],
        'height': basin_array.shape[0],
        'transform': basin_affine_tf
    })
    with rasterio.open(fp=basin_dem_file, mode='w', **out_meta) as dst:
        dst.write(basin_array, 1)

    return basin_dem_file


def calculate_hand(out_raster, dem_array, dem_affine: rasterio.Affine, dem_crs: rasterio.crs.CRS, mask,
                   acc_thresh: Optional[int] = 100):
    grid = Pgrid()
    grid.add_gridded_data(dem_array, data_name='dem', affine=dem_affine, crs=dem_crs.to_dict(), mask=mask)

    dirn = os.path.dirname(out_raster)
    prefix = os.path.basename(out_raster).split(".tif")[0]
    # Fill pits in DEM
    grid.fill_pits('dem', out_name='pit_filled_dem')

    out_by_write_cog = os.path.join(dirn, f"{prefix}_pit_filled_dem.tif")
    write_cog(out_by_write_cog, grid.pit_filled_dem, transform=dem_affine.to_gdal(), epsg_code=dem_crs.to_epsg())

    # log.info('Filling depressions')
    grid.fill_depressions('pit_filled_dem', out_name='flooded_dem')
    #if np.isnan(grid.flooded_dem).any():
    #    log.debug('NaNs encountered in flooded DEM; filling.')
    #    grid.flooded_dem = fill_nan(grid.flooded_dem)

    # output flooded_dem.tif

    out_by_write_cog = os.path.join(dirn, f"{prefix}_flooded_dem.tif")
    write_cog(out_by_write_cog, grid.flooded_dem, transform=dem_affine.to_gdal(), epsg_code=dem_crs.to_epsg())

    log.info('Resolving flats')
    grid.resolve_flats('flooded_dem', out_name='inflated_dem')
    #if np.isnan(grid.inflated_dem).any():
    #    log.debug('NaNs encountered in inflated DEM; replacing NaNs with original DEM values')
    #    grid.inflated_dem[np.isnan(grid.inflated_dem)] = dem_array[np.isnan(grid.inflated_dem)]

    # output inflated_dem
    out_by_write_cog = os.path.join(dirn, f"{prefix}_inflated_dem.tif")
    write_cog(out_by_write_cog, grid.inflated_dem, transform=dem_affine.to_gdal(), epsg_code=dem_crs.to_epsg())

    # log.info('Obtaining flow direction')
    # original apply_mask=True
    # grid.flowdir(data='inflated_dem', out_name='dir', apply_mask=True)
    grid.flowdir(data='inflated_dem', out_name='dir', apply_mask=False)
    # if np.isnan(grid.dir).any():
    #    log.debug('NaNs encountered in flow direction; filling.')
    #    grid.dir = fill_nan(grid.dir)
    # output dir
    out_by_write_cog = os.path.join(dirn, f"{prefix}_dir.tif")
    write_cog(out_by_write_cog, grid.dir, transform=dem_affine.to_gdal(), epsg_code=dem_crs.to_epsg())

    #log.info('Calculating flow accumulation')
    grid.accumulation(data='dir', out_name='acc')
    #if np.isnan(grid.acc).any():
    #    log.debug('NaNs encountered in accumulation; filling.')
    #    grid.acc = fill_nan(grid.acc)
    # output acc
    out_by_write_cog = os.path.join(dirn, f"{prefix}_acc.tif")
    write_cog(out_by_write_cog, grid.acc, transform=dem_affine.to_gdal(), epsg_code=dem_crs.to_epsg())

    if acc_thresh is None:
        acc_thresh = grid.acc.mean()

    #log.info(f'Calculating HAND using accumulation threshold of {acc_thresh}')
    hand = grid.compute_hand('dir', 'inflated_dem', grid.acc > acc_thresh, inplace=False)

    # output hand
    out_by_write_cog = os.path.join(dirn, f"{prefix}_hand.tif")
    write_cog(out_by_write_cog, hand, transform=dem_affine.to_gdal(), epsg_code=dem_crs.to_epsg())


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


    with fiona.open(args.vector_file) as vds:
        geometries = GeometryCollection([shape(feature['geometry']) for feature in vds])

    with NamedTemporaryFile(suffix='.vrt', delete=False) as dem_vrt:
        prepare_dem_vrt(dem_vrt.name, geometries)

    with rasterio.open(dem_vrt.name) as src:
        basin_mask, basin_affine_tf, basin_window = rasterio.mask.raster_geometry_mask(
            src, geometries, all_touched=True, crop=True, pad=True, pad_width=1
        )

        basin_array = src.read(1, window=basin_window)

        dirn = os.path.dirname(args.out_raster)
        prefix = os.path.basename(args.out_raster).split(".tif")[0]
        basin_dem_file = os.path.join(dirn, f"{prefix}_dem.tif")
        get_basin_dem_file(src, basin_affine_tf, basin_array, basin_dem_file)

        calculate_hand(args.out_raster, basin_array, basin_affine_tf, src.crs, ~basin_mask)

    print("completed...")


if __name__ == "__main__":

    main()
