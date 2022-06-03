"""Calculate Height Above Nearest Drainage (HAND) from the Copernicus GLO-30 DEM"""
import argparse
import logging
import sys
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Union

import astropy.convolution
import fiona
import numpy as np
import rasterio.crs
import rasterio.mask
from pysheds.sgrid import sGrid
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


def fill_hand(hand: np.ndarray, dem: np.ndarray):
    """Replace NaNs in a HAND array with values interpolated from their neighbor's HOND

    Replace NaNs in a HAND array with values interpolated from their neighbor's HOND (height of nearest drainage)
    using a 2D Gaussian kernel. Here, HOND is defined as the DEM value less the HAND value. For the kernel, see:
    https://docs.astropy.org/en/stable/convolution/#using-astropy-s-convolution-to-replace-bad-data
    """
    hond = dem - hand
    hond = fill_nan(hond)

    hand_mask = np.isnan(hand)
    hand[hand_mask] = dem[hand_mask] - hond[hand_mask]
    hand[hand < 0] = 0

    return hand


def calculate_hand(dem_file: Union[str, Path], acc_thresh: Optional[int] = 100):
    """Calculate the Height Above Nearest Drainage (HAND)

     Calculate the Height Above Nearest Drainage (HAND) using pySHEDS library. Because HAND
     is tied to watershed boundaries (hydrobasins), clipped/cut basins will produce weird edge
     effects, and incomplete basins should be masked out. For watershed boundaries,
     see: https://www.hydrosheds.org/page/hydrobasins

     This involves:
        * Filling pits (single-cells lower than their surrounding neighbors)
            in the Digital Elevation Model (DEM)
        * Filling depressions (regions of cells lower than their surrounding neighbors)
            in the Digital Elevation Model (DEM)
        * Resolving un-drainable flats
        * Determine the flow direction using the ESRI D8 routing scheme
        * Determine flow accumulation (number of upstream cells)
        * Create a drainage mask using the accumulation threshold `acc_thresh`
        * Calculating HAND

    In the HAND calculation, NaNs inside the basin filled using `fill_hand`

    Args:
        dem_file: Path to DEM raster to calculate HAND for
        acc_thresh: Accumulation threshold for determining the drainage mask.
            If `None`, the mean accumulation value is used
    """
    # From PySheds; see example usage: http://mattbartos.com/pysheds/
    grid = sGrid.from_raster(str(dem_file))
    dem = grid.read_raster(str(dem_file))

    log.info('Fill pits in DEM')
    pit_filled_dem = grid.fill_pits(dem)

    log.info('Filling depressions')
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    del pit_filled_dem

    log.info('Resolving flats')
    inflated_dem = grid.resolve_flats(flooded_dem)
    del flooded_dem

    log.info('Obtaining flow direction')
    flow_dir = grid.flowdir(inflated_dem, apply_mask=True)

    log.info('Calculating flow accumulation')
    acc = grid.accumulation(flow_dir)

    if acc_thresh is None:
        acc_thresh = acc.mean()

    log.info(f'Calculating HAND using accumulation threshold of {acc_thresh}')
    hand = grid.compute_hand(flow_dir, inflated_dem, acc > acc_thresh, inplace=False)

    log.info('Setting lowland flats (rivers) to zero')
    if np.isnan(hand).any():
        valid_mask = ~np.isnan(hand)
        mean_height = inflated_dem[valid_mask].mean()

        # calculate gradient and set mean gradient magnitude as threshold for flatness.
        g0, g1 = np.gradient(inflated_dem)
        g_mag = np.sqrt(g0 ** 2 + g1 ** 2)

        g_mag[valid_mask] = np.nan
        g_mag_threshold = np.min([1, np.nanmean(g_mag)])
        # TODO: combine next two lines
        valid_flats = np.logical_and(~valid_mask, g_mag < g_mag_threshold)
        valid_low_flats = np.logical_and(valid_flats, inflated_dem < mean_height)
        hand[valid_low_flats] = 0

    return hand


def calculate_hand_for_basins(out_raster:  Union[str, Path], geometries: GeometryCollection,
                              dem_file: Union[str, Path]):
    """Calculate the Height Above Nearest Drainage (HAND) for watershed boundaries (hydrobasins).

    For watershed boundaries, see: https://www.hydrosheds.org/page/hydrobasins

    Args:
        out_raster: HAND GeoTIFF to create
        geometries: watershed boundary (hydrobasin) polygons to calculate HAND over
        dem_file: DEM raster covering (containing) `geometries`
    """
    nodata_fill_value = np.finfo(float).eps

    with rasterio.open(dem_file) as src:
        basin_mask, basin_affine_tf, basin_window = rasterio.mask.raster_geometry_mask(
            src, geometries, all_touched=True, crop=True, pad=True, pad_width=1
        )

        basin_array = src.read(1, window=basin_window)

        with NamedTemporaryFile() as temp_file:
            write_cog(temp_file.name, basin_array,
                      transform=basin_affine_tf.to_gdal(), epsg_code=src.crs.to_epsg(),
                      # Prevents PySheds from assuming using zero as the nodata value
                      nodata_value=nodata_fill_value)

            hand = calculate_hand(temp_file.name)

        # mask outside of basin with a not-NaN value to prevent NaN-filling outside of basin (optimization)
        hand[basin_mask] = nodata_fill_value

        if np.isnan(hand).any():
            hand = fill_hand(hand, basin_array)

        # TODO: also mask ocean pixels here?

        # FIXME: what is the right nodata value here?
        # fill basin_mask with nan
        hand[basin_mask] = np.nan

        write_cog(str(out_raster), hand,
                  transform=basin_affine_tf.to_gdal(), epsg_code=src.crs.to_epsg(),
                  nodata_value=nodata_fill_value,
                  )


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
