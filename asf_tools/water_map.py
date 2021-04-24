"""Generate surface water maps from Sentinel-1 RTC products"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from osgeo import gdal

from asf_tools.composite import get_epsg_code, write_cog
from asf_tools.raster import read_as_masked_array
from asf_tools.threshold import expectation_maximization_threshold as em_threshold
from asf_tools.util import tile_array, untile_array

log = logging.getLogger(__name__)

VV_DEFAULT_THRESHOLD = -17. / 10. + 30.  # db -> db-like
VH_DEFAULT_THRESHOLD = -24. / 10. + 30.  # db -> db-like


def std_of_subtiles(tiles: np.ndarray) -> np.ndarray:
    sub_tile_shape = (tiles.shape[1] // 2, tiles.shape[2] // 2)
    sub_tiles_std = np.zeros((tiles.shape[0], 4))
    for ii, tile in enumerate(tiles):
        sub_tiles = tile_array(tile, tile_shape=sub_tile_shape)
        sub_tiles_std[ii, :] = sub_tiles.std(axis=(1, 2))
    return sub_tiles_std


def mean_of_subtiles(tiles: np.ndarray) -> np.ndarray:
    sub_tile_shape = (tiles.shape[1] // 2, tiles.shape[2] // 2)
    sub_tiles_mean = np.zeros((tiles.shape[0], 4))
    for ii, tile in enumerate(tiles):
        sub_tiles = tile_array(tile.filled(0), tile_shape=sub_tile_shape)
        sub_tiles_mean[ii, :] = sub_tiles.mean(axis=(1, 2))
    return sub_tiles_mean


def select_hand_tiles(tiles: np.ndarray, hand_threshold: float, hand_fraction: float) -> np.ndarray:
    tile_indexes = np.arange(tiles.shape[0])

    tiles = np.ma.masked_greater_equal(tiles, hand_threshold)
    percent_valid_pixels = np.sum(~tiles.mask, axis=(1, 2)) / (tiles.shape[1] * tiles.shape[2])

    return tile_indexes[percent_valid_pixels > hand_fraction]


def select_backscatter_tiles(backscatter_tiles: np.ndarray, hand_candidates: np.ndarray) -> np.ndarray:
    tile_indexes = np.arange(backscatter_tiles.shape[0])

    sub_tile_means = mean_of_subtiles(backscatter_tiles)
    sub_tile_means_std = sub_tile_means.std(axis=1)
    tile_medians = np.ma.median(backscatter_tiles, axis=(1, 2))
    tile_variance = sub_tile_means_std / tile_medians  # OK

    low_mean_threshold = np.ma.median(tile_medians[hand_candidates])
    low_mean_candidates = tile_indexes[tile_medians < low_mean_threshold]

    potential_candidates = np.intersect1d(hand_candidates, low_mean_candidates)

    for variance_threshold in np.percentile(tile_variance, np.arange(5, 96)[::-1]):
        variance_candidates = tile_indexes[tile_variance > variance_threshold]
        selected = np.intersect1d(variance_candidates, potential_candidates)
        sort_index = np.argsort(sub_tile_means_std[selected])[::-1]
        if len(selected) >= 5:
            return selected[sort_index][:5]


def determine_em_threshold(tiles: np.ndarray, scaling: float) -> float:
    thresholds = []
    for ii in range(tiles.shape[0]):
        test_tile = (np.around(tiles[ii, :, :] * scaling)).astype(int)
        thresholds.append(em_threshold(test_tile) / scaling)

    return np.median(np.sort(thresholds)[:4])


def make_water_map(out_raster: Union[str, Path], primary: Union[str, Path], secondary: Union[str, Path],
                   hand: Union[str, Path], tile_shape: Tuple[int, int] = (100, 100),
                   hand_threshold: float = 15., hand_fraction: float = 0.8):
    """Creates a surface water extent map from a Sentinel-1 RTC product

    Args:
        out_raster: Water map GeoTIFF to create
        primary: Sentinel-1 RTC GeoTIFF raster, in power scale, of the primary polarization
        secondary: Sentinel-1 RTC GeoTIFF raster, in power scale, of the secondary polarization
        hand: Height Above Nearest Drainage (HAND) GeoTIFF aligned to the rasters
        tile_shape:
        hand_threshold:
        hand_fraction:
    """
    if tile_shape[0] % 2 or tile_shape[1] % 2:
        raise ValueError(f'tile_shape {tile_shape} requires even values.')

    hand_array = read_as_masked_array(str(hand))

    hand_tiles = tile_array(hand_array, tile_shape=tile_shape, pad_value=np.nan)
    hand_candidates = select_hand_tiles(hand_tiles, hand_threshold, hand_fraction)

    log.info('Creating initial water mask from primary raster')
    primary_array = read_as_masked_array(str(primary))
    # Masking less than zero only necessary for old HyP3/GAMMA products which sometimes returned negative powers
    primary_tiles = np.ma.masked_less_equal(tile_array(primary_array, tile_shape=tile_shape, pad_value=0.), 0.)
    selected_primary_tiles = select_backscatter_tiles(primary_tiles, hand_candidates)

    primary_tiles = np.log10(primary_tiles) + 30  # linear power distribution --> gaussian (db-like) distribution
    if selected_primary_tiles is None:
        log.warning('Tile selection did not converge! using default thresholds')
        primary_threshold = VH_DEFAULT_THRESHOLD
    else:
        primary_scaling = 256 / (np.mean(primary_tiles) + 3 * np.std(primary_tiles))
        primary_threshold = determine_em_threshold(primary_tiles[selected_primary_tiles, :, :], primary_scaling)
        primary_threshold = primary_threshold if primary_threshold < VH_DEFAULT_THRESHOLD else VH_DEFAULT_THRESHOLD

    primary_tiles = np.ma.masked_less_equal(primary_tiles, primary_threshold)
    primary_water_map = untile_array(primary_tiles.mask, primary_array.shape) & ~primary_array.mask

    log.info('Creating initial water mask from secondary raster')
    secondary_array = read_as_masked_array(str(secondary))
    # Masking less than zero only necessary for old HyP3/GAMMA products which sometimes returned negative powers
    secondary_tiles = np.ma.masked_less_equal(tile_array(secondary_array, tile_shape=tile_shape, pad_value=0.), 0.)

    secondary_tiles = np.log10(secondary_tiles) + 30  # linear power distribution --> gaussian (db-like) distribution
    if selected_primary_tiles is None:
        secondary_threshold = VV_DEFAULT_THRESHOLD
    else:
        secondary_scaling = 256 / (np.mean(secondary_tiles) + 3 * np.std(secondary_tiles))
        secondary_threshold = determine_em_threshold(secondary_tiles[selected_primary_tiles, :, :], secondary_scaling)
        secondary_threshold = secondary_threshold if secondary_threshold < VV_DEFAULT_THRESHOLD else VV_DEFAULT_THRESHOLD

    secondary_tiles = np.ma.masked_less_equal(secondary_tiles, secondary_threshold)
    secondary_water_map = untile_array(secondary_tiles.mask, secondary_array.shape) & ~secondary_array.mask

    log.info('Combining primary and secondary water masks')
    combined_water_map = primary_water_map | secondary_water_map

    primary_info = gdal.Info(str(primary), format='json')
    write_cog(str(out_raster), combined_water_map, transform=primary_info['geoTransform'],
              epsg_code=get_epsg_code(primary_info), dtype=gdal.GDT_Byte, nodata_value=False)
    write_cog(str(out_raster).replace('.tif', '_VH.tif'), primary_water_map, transform=primary_info['geoTransform'],
              epsg_code=get_epsg_code(primary_info), dtype=gdal.GDT_Byte, nodata_value=False)
    write_cog(str(out_raster).replace('.tif', '_VV.tif'), secondary_water_map, transform=primary_info['geoTransform'],
              epsg_code=get_epsg_code(primary_info), dtype=gdal.GDT_Byte, nodata_value=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('out_raster', type=Path, help='Water map GeoTIFF to create')
    # FIXME: Don't assume power scale?
    parser.add_argument('primary', type=Path,
                        help='Sentinel-1 RTC GeoTIFF raster, in power scale, of the primary polarization')
    parser.add_argument('secondary', type=Path,
                        help='Sentinel-1 RTC GeoTIFF raster, in power scale, of the secondary polarization')
    # FIXME: Don't assume warped HAND
    parser.add_argument('hand', type=Path,
                        help='Height Above Nearest Drainage (HAND) GeoTIFF aligned to the rasters')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    log.debug(' '.join(sys.argv))
    log.info(f'Creating a water map from raster(s): {args.primary} {args.secondary}')

    make_water_map(args.out_raster, args.primary, args.secondary, args.hand)

    log.info(f'Water map created successfully: {args.out_raster}')
