"""Generate surface water maps from Sentinel-1 RTC products

Create a surface water extent map from a dual-pol Sentinel-1 RTC product and
a HAND image. The HAND image must be pixel-aligned (same extent and size) to
the RTC images. The water extent maps are created using an adaptive Expectation
Maximization thresholding approach.
"""

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
    return np.array([])


def determine_em_threshold(tiles: np.ndarray, scaling: float) -> float:
    thresholds = []
    for ii in range(tiles.shape[0]):
        test_tile = (np.around(tiles[ii, :, :] * scaling)).astype(int)
        thresholds.append(em_threshold(test_tile) / scaling)

    return np.median(np.sort(thresholds)[:4])


def make_water_map(out_raster: Union[str, Path], vv_raster: Union[str, Path], vh_raster: Union[str, Path],
                   hand_raster: Union[str, Path], tile_shape: Tuple[int, int] = (100, 100),
                   max_vv_threshold: float = -17., max_vh_threshold: float = -24.,
                   hand_threshold: float = 15., hand_fraction: float = 0.8):
    """Creates a surface water extent map from a Sentinel-1 RTC product

    Create a surface water extent map from a dual-pol Sentinel-1 RTC product and
    a HAND image. The HAND image must be pixel-aligned (same extent and size) to
    the RTC images. The water extent maps are created using an adaptive Expectation
    Maximization thresholding approach.

    The input images are broken into a set of corresponding tiles with a shape of
    `tile_shape`, and a set of tiles are selected from the VH RTC
    image that contain water boundaries to determine an appropriate water threshold.
     Candidate tiles must meet these criteria:
    * `hand_fraction` of pixels within a tile must have HAND pixel values lower
      than `hand_threshold`
    * The median backscatter value for the tile must be lower than an average tiles'
      backscatter values
    * The tile must have a high variance -- high variance is considered initially to
      be a variance in the 95th percentile of the tile variances, but progressively
      relaxed to the 5th percentile if there not at least 5 candidate tiles.

    The 5 VH tiles with the highest variance are selected for thresholding and a
    water threshold value is determined using an Expectation Maximization approach.
    If there were not enough candidate tiles or the threshold is too high,
    `max_vh_threshold` and/or `max_vv_threshold` will be used instead.

    Args:
        out_raster: Water map GeoTIFF to create
        vv_raster: Sentinel-1 RTC GeoTIFF, in power scale, with VV polarization
        vh_raster: Sentinel-1 RTC GeoTIFF, in power scale, with VH polarization
        hand_raster: Height Above Nearest Drainage (HAND) GeoTIFF aligned to the RTC rasters
        tile_shape: shape (height, width) in pixels to tile the image to
        max_vv_threshold: Maximum threshold value to use for `vv_raster` in decibels (db)
        max_vh_threshold:  Maximum threshold value to use for `vh_raster` in decibels (db)
        hand_threshold: The maximum height above nearest drainage in meters to consider
            a pixel valid
        hand_fraction: The minimum fraction of valid HAND pixels required in a tile for
            thresholding
    """
    if tile_shape[0] % 2 or tile_shape[1] % 2:
        raise ValueError(f'tile_shape {tile_shape} requires even values.')

    hand_array = read_as_masked_array(hand_raster)

    hand_tiles = tile_array(hand_array, tile_shape=tile_shape, pad_value=np.nan)
    hand_candidates = select_hand_tiles(hand_tiles, hand_threshold, hand_fraction)

    selected_tiles = None
    water_extent_maps = []
    for max_threshold, raster in ((max_vh_threshold, vh_raster), (max_vv_threshold, vv_raster)):
        log.info(f'Creating initial water mask from {raster}')
        array = read_as_masked_array(raster)
        tiles = tile_array(array, tile_shape=tile_shape, pad_value=0.)
        # Masking less than zero only necessary for old HyP3/GAMMA products which sometimes returned negative powers
        tiles = np.ma.masked_less_equal(tiles, 0.)
        if selected_tiles is None:
            selected_tiles = select_backscatter_tiles(tiles, hand_candidates)
            log.info(f'Selected tiles {selected_tiles} from {raster}')

        tiles = np.log10(tiles) + 30.  # linear power scale --> Gaussian (db-like) scale optimized for thresholding
        max_threshold = max_threshold / 10. + 30.  # db --> Gaussian (db-like) scale optimized for thresholding
        if selected_tiles.size:
            scaling = 256 / (np.mean(tiles) + 3 * np.std(tiles))
            threshold = determine_em_threshold(tiles[selected_tiles, :, :], scaling)
            log.info(f'Threshold determined to be {threshold}')
            if threshold > max_threshold:
                log.info(f'Threshold too high! Using maximum threshold {max_threshold}')
                threshold = max_threshold
        else:
            log.info(f'Tile selection did not converge! using default threshold {max_threshold}')
            threshold = max_threshold

        tiles = np.ma.masked_less_equal(tiles, threshold)
        water_map = untile_array(tiles.mask, array.shape) & ~array.mask

        water_extent_maps.append(water_map)

        del array, tiles

    log.info('Combining VH and VV water masks')
    combined_water_map = np.logical_or(*water_extent_maps)

    raster_info = gdal.Info(str(vh_raster), format='json')
    write_cog(out_raster, combined_water_map, transform=raster_info['geoTransform'],
              epsg_code=get_epsg_code(raster_info), dtype=gdal.GDT_Byte, nodata_value=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('out_raster', type=Path, help='Water map GeoTIFF to create')
    # FIXME: Decibel RTCs would be real nice.
    parser.add_argument('vv_raster', type=Path,
                        help='Sentinel-1 RTC GeoTIFF raster, in power scale, with VV polarization')
    parser.add_argument('vh_raster', type=Path,
                        help='Sentinel-1 RTC GeoTIFF raster, in power scale, with VH polarization')
    # FIXME: Don't assume pixel-aligned HAND
    parser.add_argument('hand_raster', type=Path,
                        help='Height Above Nearest Drainage (HAND) GeoTIFF aligned to the RTC rasters')

    parser.add_argument('--tile-shape', type=int, nargs=2, default=(100, 100),
                        help='shape (height, width) in pixels to tile the image to')
    parser.add_argument('--max-vv-threshold', type=float, default=-17.,
                        help='Maximum threshold value to use for `vv_raster` in decibels (db)')
    parser.add_argument('--max-vh-threshold', type=float, default=-25.,
                        help='Maximum threshold value to use for `vh_raster` in decibels (db)')
    parser.add_argument('--hand-threshold', type=float, default=15.,
                        help='The maximum height above nearest drainage in meters to consider a pixel valid')
    parser.add_argument('--hand-fraction', type=float, default=0.8,
                        help='The minimum fraction of valid HAND pixels required in a tile for thresholding')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    log.debug(' '.join(sys.argv))

    make_water_map(args.out_raster, args.vv_raster, args.vh_raster, args.hand_raster, args.tile_shape,
                   args.max_vv_threshold, args.max_vh_threshold, args.hand_threshold, args.hand_fraction)

    log.info(f'Water map created successfully: {args.out_raster}')
