"""Calculate Height Above Nearest Drainage (HAND) from the Copernicus GLO-30 DEM"""
import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
import urllib
from typing import Optional, Union
import zipfile

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
        print(f"Number of shapes loaded: {len(shapes)}")
        if get_property is not None:
            props = [feature["properties"][get_property] for feature in shpf]
            return shapes, props
        else:
            return shapes


def calculate_hand(dem_array, dem_affine: rasterio.Affine, dem_crs: rasterio.crs.CRS, basin_mask,
                   acc_thresh: Optional[int] = 100):
    """Calculate the Height Above Nearest Drainage (HAND)

     Calculate the Height Above Nearest Drainage (HAND) using pySHEDS library. Because HAND
     is tied to watershed boundaries (hydrobasins), clipped/cut basins will produce weird edge
     effects, and incomplete basins should be masked out. For watershed boundaries,
     see: https://www.hydrosheds.org/page/hydrobasins

     This involves:
        * Filling depressions (regions of cells lower than their surrounding neighbors)
            in the Digital Elevation Model (DEM)
        * Resolving un-drainable flats
        * Determine the flow direction using the ESRI D8 routing scheme
        * Determine flow accumulation (number of upstream cells)
        * Create a drainage mask using the accumulation threshold `acc_thresh`
        * Calculating HAND

    In the HAND calculation, NaNs inside the basin filled using `fill_nan`

    Args:
        dem_array: DEM to calculate HAND for
        dem_crs: DEM Coordinate Reference System (CRS)
        dem_affine: DEM Affine geotransform
        basin_mask: Array of booleans indicating wither an element should be masked out (à la Numpy Masked Arrays:
            https://numpy.org/doc/stable/reference/maskedarray.generic.html#what-is-a-masked-array)
        acc_thresh: Accumulation threshold for determining the drainage mask.
            If `None`, the mean accumulation value is used
    """

    grid = Pgrid()
    grid.add_gridded_data(dem_array, data_name='dem', affine=dem_affine, crs=dem_crs.to_dict(), mask=~basin_mask)

    log.info('Filling depressions')
    grid.fill_depressions('dem', out_name='flooded_dem')
    if np.isnan(grid.flooded_dem).any():
        log.debug('NaNs encountered in flooded DEM; filling.')
        grid.flooded_dem = fill_nan(grid.flooded_dem)

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
    # hand[basin_mask] = np.nan

    return hand


def get_land_mask(hand, dem):
    nodata_fill_value = np.finfo(float).eps
    # generate nan_mask
    nan_mask = np.isnan(hand)
    # Download GSHHG
    gshhg_dir = '/media/jzhu4/data/hand/external_data'
    gshhg_url = 'http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip'
    gshhg_zipfile = os.path.join(gshhg_dir, "gshhg-shp-2.3.7.zip")
    gshhg_file = os.path.join(gshhg_dir, "GSHHS_shp/f/GSHHS_f_L1.shp")
    if not os.path.exists(gshhg_zipfile) and not os.path.exists(gshhg_file):
        urllib.request.urlretrieve(gshhg_url, gshhg_zipfile)
    if not os.path.exists(gshhg_file):
        with zipfile.ZipFile(gshhg_zipfile, 'r') as zip_ref:
            zip_ref.extractall(path=gshhg_dir)

    gshhg = fiona_read_vectorfile(gshhg_file)
    # generate land_mask for the DEM
    land_mask, tf, win = rasterio.mask.raster_geometry_mask(dem, gshhg, crop=False,
                                                            invert=True)
    # invert=If False (default), mask will be False inside shapes and True outside
    # set ocean/sea values in hand to epsilon
    hand[np.invert(land_mask)] = nodata_fill_value  # sea_mask=np.invert(land_mask)
    # find nan areas that are within land_mask
    joint_mask = np.bitwise_and(nan_mask, land_mask)
    mask_labels, num_labels = ndimage.label(joint_mask)
    print(f"Number of NaN areas to fill: {num_labels}")

    return hand, mask_labels, num_labels, joint_mask


def fill_data_with_nan(hand, dem, mask_labels, num_labels, joint_mask):
    # new nan_fill needs DEM. Might be better to NOT load it in the memory
    # See: https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
    demarray = dem.read(1)
    object_slices = ndimage.find_objects(mask_labels)
    tq = range(1, num_labels)
    for lb in tq:
        slices = object_slices[lb - 1]
        min0 = max(slices[0].start - 1, 0)
        max0 = min(slices[0].stop + 1, mask_labels.shape[0])
        min1 = max(slices[1].start - 1, 0)
        max1 = min(slices[1].stop + 1, mask_labels.shape[1])
        mask_labels_clip = mask_labels[min0:max0, min1:max1]
        h = hand[min0:max0, min1:max1]
        d = demarray[min0:max0, min1:max1]
        m = joint_mask[min0:max0, min1:max1].copy()
        m[mask_labels_clip != lb] = 0
        hf = fill_nan_based_on_dem(h.copy(), d.copy())
        h[m] = hf[m]

    return hand


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

        hand = calculate_hand(basin_array, basin_affine_tf, src.crs, basin_mask)
        # fill non basin_mask with nodata_fill_value
        nodata_fill_value = np.finfo(float).eps
        hand[basin_mask] = nodata_fill_value
        if np.isnan(hand).any():
            basin_dem_file = "/tmp/tmp_dem.tif"
            get_basin_dem_file(src, basin_affine_tf, basin_array, basin_dem_file)
            # get joint_mask which is the nan_mask and the land_mask.
            # fill ocean pixels with minimum value of the float32.
            basin_dem = rasterio.open(basin_dem_file, 'r')
            hand, mask_labels, num_labels, joint_mask = get_land_mask(hand, basin_dem)
            # fill nan pixels
            hand = fill_data_with_nan(hand, basin_dem, mask_labels, num_labels, joint_mask)
            tf_gdal = basin_dem.meta['transform'].to_gdal()
            epsg_code = basin_dem.crs.to_epsg()
        else:
            tf_gdal = basin_affine_tf.to_gdal()
            epsg_code = src.crs.to_epsg()

        # fill non basin_mask with nan
        hand[basin_mask] = np.nan
        # write the HAND
        write_cog(str(out_raster), hand, transform=tf_gdal, epsg_code=epsg_code)


'''
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

        hand = calculate_hand(basin_array, basin_affine_tf, src.crs, basin_mask)

        write_cog(str(out_raster), hand, transform=basin_affine_tf.to_gdal(), epsg_code=src.crs.to_epsg())
'''

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
