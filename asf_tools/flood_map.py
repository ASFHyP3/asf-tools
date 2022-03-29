
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from asf_tools.composite import write_cog
from osgeo import gdal
from scipy import ndimage, optimize, stats

log = logging.getLogger(__name__)


def check_coordinate_system(info):
    info = info['coordinateSystem']['wkt']
    return info.split('ID')[-1].split(',')[1].replace(']', '')


def get_coordinates(info):
    west, south = info['cornerCoordinates']['lowerLeft']
    east, north = info['cornerCoordinates']['upperRight']
    return west, south, east, north


def get_waterbody(input_info, ths=30.):
    epsg_code = 'EPSG:' + check_coordinate_system(input_info)

    west, south, east, north = get_coordinates(input_info)
    width, height = input_info['size']

    gitdir = os.path.dirname(__file__)
    water_extent_vrt = str(str(gitdir) + '/data/water_extent.vrt')  # All Perennial Flood Data
    wimage_file = str(str(gitdir) + '/data/surface_water_map_clip.tif')

    gdal.Warp(wimage_file, water_extent_vrt, dstSRS=epsg_code,
              outputBounds=[west, south, east, north],
              width=width, height=height, resampleAlg='lanczos', format='GTiff')

    wimage = gdal.Open(wimage_file, gdal.GA_ReadOnly).ReadAsArray()
    return wimage > ths  # higher than 30% possibility (present water)


def iterative(hand, extent, water_levels=range(15)):
    def _goal_ts(w):
        iterative_flood_extent = hand < w  # w=water level
        tp = np.nansum(np.logical_and(iterative_flood_extent == 1, extent == 1))  # true positive
        fp = np.nansum(np.logical_and(iterative_flood_extent == 1, extent == 0))  # False positive
        fn = np.nansum(np.logical_and(iterative_flood_extent == 0, extent == 1))  # False negative
        return 1 - tp / (tp + fp + fn)  # threat score #we will minimize goal func, hence 1-threat_score.

    class MyBounds(object):
        def __init__(self, xmax=tuple(max(water_levels)), xmin=tuple(min(water_levels))):
            self.xmax = np.array(xmax)
            self.xmin = np.array(xmin)

        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            return tmax and tmin

    bounds = MyBounds()
    x0 = [np.mean(water_levels)]
    opt_res = optimize.basinhopping(_goal_ts, x0, niter=10000, niter_success=100, accept_test=bounds)
    if opt_res.message[0] == 'success condition satisfied' \
            or opt_res.message[0] == 'requested number of basinhopping iterations completed successfully':
        best_water_level = opt_res.x[0]
    else:
        best_water_level = np.inf  # unstable solution.
    return best_water_level


def logstat(data, func=np.nanstd):
    """ stat=logstat(data, func=np.nanstd)
       calculates the statistic after taking the log and returns the statistic in linear scale.
       INF values inside the data array are set to nan.
       The func has to be able to handle nan values.
    """
    ld = np.log(data)
    ld[np.isinf(ld)] = np.nan
    st = func(ld)
    return np.exp(st)


def estimate_flood_depth(label, hand, flood_labels, estimator='nmad', water_level_sigma=3., iterative_bounds=(0, 15)):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')

        if estimator.lower() == "iterative":
            return iterative(hand, flood_labels == label, water_levels=iterative_bounds)

        if estimator.lower() == "numpy":
            hand_mean = np.nanmean(hand[flood_labels == label])
            hand_std = np.nanstd(hand[flood_labels == label])

        elif estimator.lower() == "nmad":
            hand_mean = np.nanmean(hand[flood_labels == label])
            hand_std = stats.median_abs_deviation(hand[flood_labels == label], scale='normal',
                                                  nan_policy='omit')
        elif estimator.lower() == "logstat":
            hand_mean = logstat(hand[flood_labels == label], func=np.nanmean)
            hand_std = logstat(hand[flood_labels == label])

        else:
            raise ValueError(f'Unknown flood depth estimator {estimator}')

    return hand_mean + water_level_sigma * hand_std


def make_flood_map(out_raster: Union[str, Path], water_raster: Union[str, Path],
                   hand_raster: Union[str, Path], estimator: str = 'nmad',
                   water_level_sigma: float = 3.,
                   known_water_threshold: float = 30.,
                   iterative_bounds: Tuple[int, int] = (0, 15)):
    """Creates a flood depth map from a water extent map.

    Create a flood depth map from a single Hyp3-generated surface water extent map and
    a HAND image. The HAND image must completely cover the water extent map.

    Known perennial Global Surface water data are produced under the Copernicus Programme,
    and are pulled to ensure this information is accounted for in the Flood Depth Map calculation.
    This is added to the SAR-derived surface water detection maps to generate the
    final Flood Depth product.

    Flood depth maps are estimated using one of the approaches:
    *Iterative: Basin hopping optimization method to match flooded areas to flood depth
    estimates given by the HAND layer. This is the most accurate method, but also the
    most time-intensive.
    *Normalized Median Absolute Deviation (nmad): (Default) Uses a median operator to estimate
    the variation to increase robustness in the presence of outliers.
    *Logstat: Calculates the mean and standard deviation of HAND heights in the logarithmic
    domain to improve robustness for very non-Gaussian data distributions.
    *Numpy: Calculates statistics in a linear scale. Least robust to outliers and non-Ga      ussian
    distributions.

    """

    info = gdal.Info(str(water_raster), format='json')
    epsg = check_coordinate_system(info)
    geotransform = info['geoTransform']

    hand_array = gdal.Open(str(hand_raster), gdal.GA_ReadOnly).ReadAsArray()

    log.info('Fetching perennial flood data.')
    known_water_mask = get_waterbody(info, ths=known_water_threshold)

    water_map = gdal.Open(water_raster).ReadAsArray()
    flood_mask = np.bitwise_or(water_map, known_water_mask)

    labeled_flood_mask, num_labels = ndimage.label(flood_mask)
    object_slices = ndimage.find_objects(labeled_flood_mask)
    log.info(f'Detected {num_labels} water bodies...')

    flood_depth = np.zeros(flood_mask.shape)

    for ll in range(1, num_labels):  # Skip first, largest label.
        slices = object_slices[ll - 1]
        min0, max0 = slices[0].start, slices[0].stop
        min1, max1 = slices[1].start, slices[1].stop

        flood_window = labeled_flood_mask[min0:max0, min1:max1]
        hand_window = hand_array[min0:max0, min1:max1]

        water_height = estimate_flood_depth(ll, hand_window, flood_window, estimator=estimator,
                                            water_level_sigma=water_level_sigma, iterative_bounds=iterative_bounds)

        flood_depth_window = flood_depth[min0:max0, min1:max1]
        flood_depth_window[flood_window == ll] = water_height - hand_window[flood_window == ll]

    flood_depth[flood_depth < 0] = 0

    write_cog(str(out_raster).replace('.tif', f'_{estimator}_WaterDepth.tif'), flood_depth, transform=geotransform,
              epsg_code=int(epsg), dtype=gdal.GDT_Byte, nodata_value=False)
    write_cog(str(out_raster).replace('.tif', f'_{estimator}_FloodMask.tif'), flood_mask, transform=geotransform,
              epsg_code=int(epsg), dtype=gdal.GDT_Byte, nodata_value=False)

    flood_mask[known_water_mask] = 0
    flood_depth[np.bitwise_not(flood_mask)] = 0

    write_cog(str(out_raster).replace('.tif', f'_{estimator}_FloodDepth.tif'), flood_depth, transform=geotransform,
              epsg_code=int(epsg), dtype=gdal.GDT_Byte, nodata_value=False)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--out-raster',
                        help='File flood depth map will be saved to.')
    parser.add_argument('--water-extent-map',
                        help='Hyp3-Generated water extent raster file.')
    parser.add_argument('--hand-raster',
                        help='Height Above Nearest Drainage (HAND) GeoTIFF aligned to the RTC rasters. '
                             'If not specified, HAND data will be extracted from a Copernicus GLO-30 DEM based HAND.')

    parser.add_argument('--estimator', type=str, default='nmad', choices=['iterative', 'logstat', 'nmad', 'numpy'],
                        help='Flood depth estimation approach.')
    parser.add_argument('--water-level-sigma', type=float, default=3.,
                        help='Estimate max water height for each object.')
    parser.add_argument('--known-water-threshold', type=float, default=30.,
                        help='Threshold for extracting known water area in percent')
    parser.add_argument('--iterative-bounds', type=float, default=[0, 15],
                        help='.')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    log.debug(' '.join(sys.argv))

    make_flood_map(args.out_raster, args.water_extent_map, args.hand_raster, args.estimator, args.water_level_sigma,
                   args.known_water_threshold, args.iterative_bounds)

    log.info(f"Flood Map written to {args.out_raster}.")
