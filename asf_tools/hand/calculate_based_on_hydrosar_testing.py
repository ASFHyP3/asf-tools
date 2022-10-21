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
from typing import Union

import astropy.convolution
import fiona
import numpy as np
import rasterio.crs
import rasterio.mask
from osgeo import gdal, osr
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
        if get_property is not None:
            props = [feature["properties"][get_property] for feature in shpf]
            return shapes, props
        else:
            return shapes


def gdal_write(ary, geoTransform, fileformat="GTiff", filename='jupyter_rocks.tif', data_format=gdal.GDT_Float64,
               nodata=None, srs_wkt='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
                                    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],'
                                    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",'
                                    'NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]',
               options=["TILED=YES", "COMPRESS=LZW", "INTERLEAVE=BAND", "BIGTIFF=YES"], build_overviews=True):
    """gdal_write(ary, geoTransform, format="GTiff", filename='jupyter_rocks.tif', data_format=gdal.GDT_Float64,
     nodata=None, srs_proj4='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
    ary: 2D array.
    geoTransform: [top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution]
    format: "GTiff"
    """
    if ary.ndim == 2:
        Ny, Nx = ary.shape
        Nb = 1
    elif ary.ndim == 3:
        Ny, Nx, Nb = ary.shape
    else:
        print("Input array has to be 2D or 3D.")
        return None

    driver = gdal.GetDriverByName(fileformat)
    ds = driver.Create(filename, Nx, Ny, Nb, data_format, options)

    # ds.SetGeoTransform( ... ) # define GeoTransform tuple
    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    ds.SetGeoTransform(geoTransform)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(srs_wkt)
    ds.SetProjection(srs.ExportToWkt())
    if nodata is not None:
        ds.GetRasterBand(1).SetNoDataValue(0)
    if Nb == 1:
        ds.GetRasterBand(1).WriteArray(ary)
    else:
        for b in range(Nb):
            ds.GetRasterBand(b + 1).WriteArray(ary[:, :, b])
    if build_overviews:
        ds.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64, 128, 256])
    ds = None
    print("File written to: " + filename)


def calculate_hand(dem_array, dem_gt, dem_proj4, mask=None, verbose=False, acc_thresh=100):
    """
    hand=calculate_hand(dem, dem_gT, dem_proj4, mask=None, verbose=False)
    Calculate the height above nearest drainage using pySHEDS library. This is done over a few steps:

    Fill_Depressions fills depressions in a DEM (regions of cells lower than their surrounding neighbors).
    Resolve_Flats resolves drainable flats in a DEM.
    FlowDir converts the DEM to flow direction based on dirmap.
    Accumulation converts from flow direction to flow accumulation.
    Compute_Hand is used to convert directions to height above nearest drainage.

    NaN values are filled at the end of resolve_flats and final steps.

    Inputs:
      dem=Numpy array of Digital Elevation Model (DEM) to convert to HAND.
      dem_gt= GeoTransform of the input DEM
      dem_proj4=Proj4 string of DEM
      mask=If provided parts of DEM can be masked out. If not entire DEM is evaluated.
      verbose=If True, provides information about where NaN values are encountered.
      acc_thresh=Accumulation threshold. By default is set to 100. If none,
                 mean value of accumulation array (acc.mean()) is used.
    """

    # Specify directional mapping
    # N, NE, E, SE, S, SW, W, NW
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    if mask is None:
        mask = np.ones(dem_array.shape, dtype=np.bool)
    grid = Pgrid()
    grid.add_gridded_data(dem_array, data_name='dem', affine=dem_gt, crs=dem_proj4, mask=mask)
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

    # in the original hydrosar notebook, apply_mask=True
    grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap, apply_mask=True)

    if np.isnan(grid.dir).any():
        log.debug('NaNs encountered in flow direction; filling.')
        grid.dir = fill_nan(grid.dir)

    log.info('Calculating flow accumulation')
    grid.accumulation(data='dir', dirmap=dirmap, out_name='acc')
    if np.isnan(grid.acc).any():
        log.debug('NaNs encountered in accumulation; filling.')
        grid.acc = fill_nan(grid.acc)

    if acc_thresh is None:
        acc_thresh = grid.acc.mean()

    log.info(f'Calculating HAND using accumulation threshold of {acc_thresh}')
    hand = grid.compute_hand('dir', 'inflated_dem', grid.acc > acc_thresh, inplace=False)

    # fill rivers, in the original hydrosar, fill river occurs at each basin.

    if np.isnan(hand).any():
        # get nans inside masked area and find mean height for pixels outside the nans (but inside basin mask)
        valid_nanmask = np.logical_and(mask, np.isnan(hand))
        valid_mask = np.logical_and(mask, ~np.isnan(hand))
        mean_height = grid.inflated_dem[valid_mask].mean()

        # calculate gradient and set mean gradient magnitude as threshold for flatness.
        g0, g1 = np.gradient(grid.inflated_dem)
        gMag = np.sqrt(g0 ** 2 + g1 ** 2)
        # gMagTh = np.min([1, np.mean(gMag * np.isnan(hand))])
        gMag[~np.isnan(hand)] = np.nan
        gMagTh = np.min([1, np.nanmean(gMag)])
        valid_flats = np.logical_and(valid_nanmask, gMag < gMagTh)
        valid_low_flats = np.logical_and(valid_flats, grid.inflated_dem < mean_height)
        hand[valid_low_flats] = 0

    return hand, grid.inflated_dem


def get_hand_by_land_mask(hand, nodata_fill_value, dem):
    nan_mask = np.isnan(hand)
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
    # find nan areas that are within land_mask
    joint_mask = np.bitwise_and(nan_mask, land_mask)
    mask_labels, num_labels = ndimage.label(joint_mask)

    return hand, mask_labels, num_labels, joint_mask


def fill_data_with_nan(hand, dem, mask_labels, num_labels, joint_mask):
    demarray = dem.read(1)
    if np.any(np.isnan(hand)):
        object_slices = ndimage.find_objects(mask_labels)
        tq = range(1, num_labels)
        for lb in tq:  # Skip first, largest label.
            slices = object_slices[lb - 1]
            min0 = max(slices[0].start - 1, 0)
            max0 = min(slices[0].stop + 1, mask_labels.shape[0])
            min1 = max(slices[1].start - 1, 0)
            max1 = min(slices[1].stop + 1, mask_labels.shape[1])
            mask_labels_clip = mask_labels[min0:max0, min1:max1]
            h = hand[min0:max0, min1:max1]  # by reference
            d = demarray[min0:max0, min1:max1]
            m = joint_mask[min0:max0, min1:max1].copy()
            m[mask_labels_clip != lb] = 0  # Mask out other flooded areas (labels) for this area. Use only one label.

            hf = fill_nan_based_on_dem(h.copy(), d.copy())  # break reference
            h[m] = hf[m]  # copy nan-fill by reference

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

    with rasterio.open(dem_file, 'r') as src:
        basin_mask, basin_affine_tf, basin_window = rasterio.mask.raster_geometry_mask(
            src, geometries, all_touched=True, crop=True, pad=True, pad_width=1)
        basin_array = src.read(1, window=basin_window)

        # produce tmp_dem.tif based on basin_array and basin_affine_tf
        basin_dem_file = "/tmp/tmp_dem.tif"
        # get_basin_dem_file(src, basin_affine_tf, basin_array, basin_dem_file)
        gdal_write(basin_array, basin_affine_tf.to_gdal(), filename=basin_dem_file,
                  srs_wkt=src.crs.to_wkt(), data_format=gdal.GDT_Float32)

    nodata_fill_value = np.finfo(float).eps
    dem = rasterio.open(basin_dem_file, 'r')
    dem_nodata_value = src.nodatavals[0]

    if dem_nodata_value is None:
        print('DEM does not have a defined no-data value.')
        print('Assuming all valid pixels. If not, expect long processing times.')
        dem_nodata_value = 0

    dem_nodata_mask = dem.read(1) == dem_nodata_value

    hand = np.zeros(dem.shape)
    hand[:] = np.nan

    inflated = np.zeros(dem.shape)

    # loop over geometries to calculate the HAND for each geometry
    for k, p in enumerate(geometries.geoms):
        verbose = False
        mask, tf, win = rasterio.mask.raster_geometry_mask(dem, [p], all_touched=True, crop=True, pad=True,
                                                           pad_width=1)

        if win.width == 1 or win.height == 1:
            continue

        not_mask = np.bitwise_not(mask)

        if dem_nodata_mask[win.row_off:win.row_off + win.height, win.col_off:win.col_off + win.width].all():
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            hand_basin, inflated_basin = calculate_hand(np.squeeze(dem.read(window=win)), tf, dem.crs.to_dict(),
                                                        mask=not_mask, verbose=verbose, acc_thresh=100)

        clip_hand = hand[win.row_off:win.row_off + win.height, win.col_off:win.col_off + win.width]  # By reference
        clip_hand[not_mask] = hand_basin[not_mask]

        clip_inflated = inflated[win.row_off:win.row_off + win.height, win.col_off:win.col_off + win.width]  # By reference
        clip_inflated[not_mask] = inflated_basin[not_mask]

    # write the HAND before fill river pixels
    hand[basin_mask] = np.nan
    dirn = os.path.dirname(out_raster)
    prefix = os.path.basename(out_raster).split(".tif")[0]
    out_by_write_cog = os.path.join(dirn, f"{prefix}_hand.tif")
    write_cog(out_by_write_cog, hand, transform=basin_affine_tf.to_gdal(), epsg_code=src.crs.to_epsg())

    # do fill_nan_based_dem
    hand[dem_nodata_mask] = nodata_fill_value
    # fill non basin_mask with nodata_fill_value
    hand[basin_mask] = nodata_fill_value

    if np.isnan(hand).any():
        basin_dem = rasterio.open(basin_dem_file, 'r')
        # basin_dem_data = basin_dem.read(1)
        # fill ocean pixels with the minimum value of data type of float32
        hand, mask_labels, num_labels, joint_mask = get_hand_by_land_mask(hand, nodata_fill_value, basin_dem)

        # fill nan pixels
        # hand = fill_nan_based_on_dem(hand, basin_dem.read(1))
        hand = fill_data_with_nan(hand, dem, mask_labels, num_labels, joint_mask)

        # fill non basin_mask with nan
        hand[basin_mask] = np.nan
        # write the HAND
        # write_cog(str(out_raster), hand, transform=basin_affine_tf.to_gdal(), epsg_code=src.crs.to_epsg())
        gdal_write(hand, basin_affine_tf.to_gdal(), filename=str(out_raster),
                   srs_wkt=src.crs.to_wkt(), nodata=np.nan, data_format=gdal.GDT_Float32)


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
