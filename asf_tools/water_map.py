"""Generate surface water maps from Sentinel-1 RTC products"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from osgeo import gdal

from asf_tools.composite import get_epsg_code, read_as_array, write_cog
from asf_tools.threshold import kittler_illingworth_threshold as ki_threshold
from asf_tools.util import tile_array

log = logging.getLogger(__name__)


def water_mask(raster: np.ndarray, tile_shape: Tuple[int, int] = (200, 200),
               sub_tile_shape: Tuple[int, int] = (100, 100)):

    tiles = np.ma.masked_invalid(tile_array(raster, tile_shape=tile_shape, pad_value=np.nan))
    sub_tiles_std = np.zeros((tiles.shape[0], 4))
    for ii, tile in enumerate(tiles):
        sub_tiles = np.ma.masked_invalid(tile_array(tile, tile_shape=sub_tile_shape))
        sub_tiles_std[ii, :] = sub_tiles.std(axis=(1, 2))

    variation = sub_tiles_std.mean(axis=1)
    # tile index in order of variation, from lowest to highest
    tile_indexes_by_variation = np.argsort(variation, axis=0)

    selected = []
    thresholds = []
    for ii in tile_indexes_by_variation[::-1]:
        threshold = ki_threshold(tiles[ii, :, :].filled(0))

        # Maximum value of threshold = -10 per Martinis et al., 2015
        if threshold >= -10:
            log.debug(f'sub-scene {ii} threshold {threshold} is above maximum value; will not be used')
            continue

        selected.append(ii)
        thresholds.append(threshold)
        log.debug(f'sub-scene {ii} selected')

        if len(selected) > 4:
            break

    if np.std(thresholds) >= 5.0:
        threshold = ki_threshold(tiles[selected, :, :].filled(0))
    else:
        threshold = np.mean(thresholds)

    log.info(f'Using threshold value of {threshold}')
    return raster < threshold


def make_water_map(out_raster: Union[str, Path], primary: Union[str, Path], secondary: Union[str, Path]):
    """Creates a surface water extent map from a Sentinel-1 RTC product

    Args:
        out_raster: Water map GeoTIFF to create
        primary: Sentinel-1 RTC GeoTIFF raster of the primary polarization
        secondary: Sentinel-1 RTC GeoTIFF raster of the secondary polarization
    """

    primary_array = read_as_array(str(primary))
    primary_mask = water_mask(primary_array)

    secondary_array = read_as_array(str(secondary))
    secondary_mask = water_mask(secondary_array)

    combined_mask = primary_mask | secondary_mask

    primary_info = gdal.Info(str(primary), format='json')
    write_cog(str(out_raster), combined_mask, transform=primary_info['geoTransform'],
              epsg_code=get_epsg_code(primary_info), dtype=gdal.GDT_Byte, nodata_value=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('out_raster', type=Path, help='Water map GeoTIFF to create')
    parser.add_argument('primary', type=Path, help='Sentinel-1 RTC GeoTIFF raster of the primary polarization')
    parser.add_argument('secondary', help='Sentinel-1 RTC GeoTIFF raster of the secondary polarization')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    log.debug(' '.join(sys.argv))
    log.info(f'Creating a water map from raster(s): {args.primary} {args.secondary}')

    make_water_map(args.out_raster, args.primary, args.secondary)

    log.info(f'Water map created successfully: {args.out_raster}')
