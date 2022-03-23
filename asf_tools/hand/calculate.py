"""Calculate Height Above Nearest Drainage (HAND) from the Copernicus GLO-30 DEM"""
import argparse
import logging
import sys
import os
from functools import partial
import warnings
from pathlib import Path
import tempfile
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Optional, Union
import urllib
import zipfile

import astropy.convolution
import fiona
import fiona.crs
import numpy as np
import rasterio.crs
import rasterio.mask
from pysheds.grid import Grid
from pysheds.pgrid import Grid as Pgrid
from affine import Affine
import shapely
from shapely.geometry import GeometryCollection, shape
from tqdm.auto import tqdm
import pyproj
from osgeo import gdal, osr
import geopandas as gpd
from scipy import ndimage
from shapely.ops import transform

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
    hond = dem - arr #height of nearest drainage
    kernel = astropy.convolution.Gaussian2DKernel(x_stddev=3) #kernel x_size=8*stddev
    arr_type = hond.dtype
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while np.any(np.isnan(hond)):
            hond = astropy.convolution.interpolate_replace_nans(hond.astype(float),
                                                                kernel, convolve=astropy.convolution.convolve)
            # test only
            ch = np.isnan(hond)
            idx = np.where(ch == True)
            print(f"number of nan in hond: {idx[0].size}")

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


def fiona_write_vectorfile(shapes, vectorfile, crs=fiona.crs.from_epsg(4326), driver='ESRI Shapefile',
                           schema_type='Polygon'):
    if schema_type == 'Polygon':
        schema = {'geometry': 'Polygon',
                  'properties': {}}
    with fiona.open(vectorfile, 'w', crs=crs, driver=driver, schema=schema) as output:
        for s in shapes:
            if schema_type == 'Polygon':
                sp = shapely.geometry.Polygon(s)
            output.write({'geometry': shapely.geometry.mapping(sp), 'properties': {}})


def vectorfile_to_shapely_shape(vectorfile):
    '''
    read the vectorfile, return them as list of shapely shape
    '''
    with fiona.open(vectorfile, "r") as shpf:
        shapes = [shapely.geometry.shape(feature["geometry"]) for feature in shpf]
    return shapes


def intersect(shapes, polygon, properties=None):
    """
    polygons=intersect(shapes, polygon, properties=None)
    Returns polygons from multiple 'geometries' read by fiona.

    shapes: shapes returned by fiona_read_vectorfile()
    polygon: a single polygon to intersect with shapes
    properties: If not none, returns the property value instead of polygon geometry.
    """
    # first loop to split multi polygons to single polygons
    polygons = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k, shape in enumerate(tqdm(shapes)):
            if shape['type'] == 'MultiPolygon':
                for l, p in enumerate(shape['coordinates']):
                    s = shapely.geometry.Polygon(p[0])
                    if polygon.intersects(s) and properties is None:
                        polygons.append(s)
                    elif polygon.intersects(s) and properties is not None:
                        if np.isscalar(properties[k]):
                            polygons.append(properties[k])
                        else:
                            polygons.append(properties[k][l])

            elif shape['type'] == 'Polygon':
                s = shapely.geometry.Polygon(shape['coordinates'][0])
                if polygon.intersects(s) and properties is None:
                    polygons.append(s)
                elif polygon.intersects(s) and properties is not None:
                    polygons.append(properties[k])
    return polygons


def reproject(vector_file, output_crs, output_file=None):
    """
    output_file=reproject(vector_file, output_crs, output_file=None)
    Reprojects a given vector file to another reference frame (CRS).
    vector_file: Any vector file that can be opened with GeoPandas
    output_crs: A rasterio opened crs (e.g. dem.crs)
    output_file: if not defined, defaults to vector_file[:-4]+'_warp.shp'.
    """
    v = gpd.GeoDataFrame.from_file(vector_file)
    warp = v.to_crs(output_crs)
    if output_file is None:
        output_file = vector_file[:-4] + '_warp.shp'
    warp.to_file(output_file)
    return output_file


def transform_polygon(polygon, s_srs='EPSG:4269', t_srs='EPSG:4326'):
    shp_geom = shapely.geometry.Polygon(polygon)
    project = pyproj.Transformer.from_proj(
        pyproj.Proj(init=s_srs),  # source coordinate system
        pyproj.Proj(init=t_srs))  # destination coordinate system

    # polygon is a shapley Polygon
    return transform(project.transform, shp_geom)  # apply projection


def transform_shape(shape, s_srs='epsg:4326', t_srs='epsg:4326'):
    transformation = partial(
        pyproj.transform,
        pyproj.Proj(init=s_srs),  # source coordinate system
        pyproj.Proj(init=t_srs))  # destination coordinate system
    return shapely.ops.transform(transformation, shape)


def xy2coord(x, y, gT):
    '''
    lon,lat=xy2coord(x,y,geoTransform)
    converts pixel index to position based on geotransform.
    '''
    coord_x = gT[0] + x * gT[1] + y * gT[2]
    coord_y = gT[3] + x * gT[4] + y * gT[5]
    return coord_x, coord_y


def get_projection(filename, out_format='proj4'):
    """
    epsg_string=get_epsg(filename, out_format='proj4')
    """
    try:
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        srs = gdal.osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjectionRef())
    except:  # I am not sure if this is working for datasets without a layer. The first try block should work mostly.
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        ly = ds.GetLayer()
        if ly is None:
            print(f"Can not read projection from file:{filename}")
            return None
        else:
            srs = ly.GetSpatialRef()
    if out_format.lower() == 'proj4':
        return srs.ExportToProj4()
    elif out_format.lower() == 'wkt':
        return srs.ExportToWkt()
    elif out_format.lower() == 'epsg':
        crs = pyproj.crs.CRS.from_proj4(srs.ExportToProj4())
        return crs.to_epsg()


def gdal_get_geotransform(filename):
    '''
    [top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution]=gdal_get_geotransform('/path/to/file')
    '''
    # http://stackoverflow.com/questions/2922532/obtain-latitude-and-longitude-from-a-geotiff-file
    ds = gdal.Open(filename)
    return ds.GetGeoTransform()


def gdal_get_size(filename):
    """(width, height) = get_size(filename)
    """
    ds = gdal.Open(filename)
    width = ds.RasterXSize
    height = ds.RasterYSize
    ds = None
    return (width, height)


def gdal_bounding_box(filename):
    """
    ((lon1,lat1), (lon2,lat2), (lon3,lat3), (lon4,lat4))=bounding_box('/path/to/file')
    """
    gT = gdal_get_geotransform(filename)
    width, height = gdal_get_size(filename)
    return (xy2coord(0, 0, gT), xy2coord(width, 0, gT), xy2coord(width, height, gT), xy2coord(0, height, gT))


def gdal_write(ary, geoTransform, fileformat="GTiff", filename='jupyter_rocks.tif', data_format=gdal.GDT_Float64,
               nodata=None, srs_proj4='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs',
               options=["TILED=YES", "COMPRESS=LZW", "INTERLEAVE=BAND", "BIGTIFF=YES"], build_overviews=True):
    '''
    gdal_write(ary, geoTransform, format="GTiff", filename='jupyter_rocks.tif', data_format=gdal.GDT_Float64 nodata=None, srs_proj4='+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
    ary: 2D array.
    geoTransform: [top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution]
    format: "GTiff"
    '''
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
    srs.ImportFromProj4(srs_proj4)
    ds.SetProjection(srs.ExportToWkt());
    if nodata is not None:
        ds.GetRasterBand(1).SetNoDataValue(0);
    if Nb == 1:
        ds.GetRasterBand(1).WriteArray(ary)
    else:
        for b in range(Nb):
            ds.GetRasterBand(b + 1).WriteArray(ary[:, :, b])
    if build_overviews:
        ds.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64, 128, 256])
    ds = None
    print("File written to: " + filename);


def point_coordinates_to_geometry(coordinates, geometry_type='Polygon'):
    if geometry_type.lower() == 'polygon':
      return shapely.geometry.Polygon(coordinates)
    else:
      raise NotImplementedError


def fill_nan(arr):
    """
    filled_arr=fill_nan(arr)
    Fills Not-a-number values in arr using astropy.
    """
    kernel = astropy.convolution.Gaussian2DKernel(x_stddev=3)  # kernel x_size=8*stddev
    arr_type = arr.dtype
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while np.any(np.isnan(arr)):
            arr = astropy.convolution.interpolate_replace_nans(arr.astype(float), kernel,
                                                               convolve=astropy.convolution.convolve)
    return arr.astype(arr_type)


def get_tight_dem(dem_vrt_name, shpfile):
    # clip the dem_vrt according to geometries
    dem = rasterio.open(dem_vrt_name)
    # shp = fiona.open(shpfile)
    # shp_crs = shp.crs
    with tempfile.TemporaryDirectory() as tmp_dir:
        shpfile1 = shpfile
        if dem.crs.to_string() != 'EPSG:4326':
            shpfile1 = reproject(shpfile, dem.crs, output_file=os.path.join(tmp_dir,'shp.shp'))

        vds = fiona.open(shpfile1)
        geometries = GeometryCollection([shape(feature['geometry']) for feature in vds])
        bounds = geometries.bounds
        os.system(f'gdal_translate {dem_vrt_name} {os.path.join(tmp_dir,"dem.tif")}')
        outfile = "out_dem.tif"
        gdal.Warp(outfile, os.path.join(tmp_dir, "dem.tif"), outputBounds=list(bounds))
    return outfile


def calculate_hand_orig(dem_array, dem_affine: rasterio.Affine, dem_proj4, basin_mask,
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
    grid.add_gridded_data(dem_array, data_name='dem', affine=dem_affine, crs=dem_proj4, mask=~basin_mask)

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
    hand[basin_mask] = np.nan

    return hand


def calculate_hand(dem_array, dem_gT, dem_proj4, mask=None, verbose=False, acc_thresh=100):
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
      dem_gT= GeoTransform of the input DEM
      dem_proj4=Proj4 string of DEM
      mask=If provided parts of DEM can be masked out. If not entire DEM is evaluated.
      verbose=If True, provides information about where NaN values are encountered.
      acc_thresh=Accumulation threshold. By default is set to 100. If none,
                 mean value of accumulation array (acc.mean()) is used.
    """

    # Specify  directional mapping
    # N , NE , E ,SE,S,SW, W , NW
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    # Load DEM into pySheds
    if type(dem_gT) == Affine:
        aff = dem_gT
    else:
        aff = Affine.from_gdal(*tuple(dem_gT))
    if mask is None:
        mask = np.ones(dem_array.shape, dtype=np.bool)
    grid = Pgrid()  # (shape=dem.shape,affine=aff, crs=dem_proj4, mask=mask)
    grid.add_gridded_data(dem_array, data_name='dem', affine=aff, crs=dem_proj4, mask=mask)
    # Fill Depressions
    grid.fill_depressions('dem', out_name='flooded_dem')
    if verbose:
        set_trace()
    if np.any(np.isnan(grid.flooded_dem)):
        if verbose:
            print('NaN:fill_depressions')
            grid.flooded_dem = fill_nan(grid.flooded_dem)
            # Resolve_Flats
    # Please note that Resolve_Flats currently has an open bug and can fail on occasion. https://github.com/mdbartos/pysheds/issues/118
    try:
        grid.resolve_flats('flooded_dem', out_name='inflated_dem')
    except:
        grid.inflated_dem = grid.flooded_dem
    # if np.sum(np.isnan(grid.inflated_dem))<dem.size*0.5: #if nans account for less than 50% of the dem nanfill.
    #    if verbose:
    #        print('NaN:resolve_flats but less than 50%. Applying large value')
    #    grid.inflated_dem=fill_nan(grid.inflated_dem)
    if np.any(np.isnan(grid.inflated_dem)):
        if verbose:
            print('NaN:resolve_flats replacing with inflated_dem')
        # grid.inflated_dem=fill_nan(grid.inflated_dem)
        grid.inflated_dem[np.isnan(grid.inflated_dem)] = dem_array[
            np.isnan(grid.inflated_dem)]  # 10000  # setting nan to 10.000 to ensure drainage
        ### Ref: https://github.com/mdbartos/pysheds/issues/90
    # Obtain flow direction
    grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap, apply_mask=True)
    if np.any(np.isnan(grid.dir)):
        if verbose:
            print('NaN:flowdir')
            grid.dir = fill_nan(grid.dir)
            # Obtain accumulation
    grid.accumulation(data='dir', dirmap=dirmap, out_name='acc')
    if np.any(np.isnan(grid.acc)):
        if verbose:
            print('NaN:accumulation')
            grid.acc = fill_nan(grid.acc)
            # Generate HAND
    if acc_thresh is None:
        acc_thresh = grid.acc.mean()
    # grid.compute_hand('dir', 'inflated_dem', grid.acc >100, out_name='hand')
    # Copy HAND as an array.
    # hand=grid.view('hand')
    hand = grid.compute_hand('dir', 'inflated_dem', grid.acc > acc_thresh, inplace=False)
    if np.any(np.isnan(hand)):
        if verbose:
            print('NaN:compute_hand')
            # attempt to fill low-lying flat areas with zeros. In radar DEMs vegetation alongside river, can trap
            # the river and not let any water go into the river. This was seen in Bangladesh with SRTM 1 arcsec
            # and NASADEM at Hydro Basin with ID: 4120928640

            # get nans inside masked area and find mean height for pixels outside the nans (but inside basin mask)
            valid_nanmask = np.logical_and(mask, np.isnan(hand))
            valid_mask = np.logical_and(mask, ~np.isnan(hand))
            mean_height = grid.inflated_dem[valid_mask].mean()
            # calculate gradient and set mean gradient magnitude as threshold for flatness.
            g0, g1 = np.gradient(grid.inflated_dem);
            gMag = np.sqrt(g0 ** 2 + g1 ** 2)
            gMagTh = np.min(1, np.mean(gMag * np.isnan(
                hand)))  # Make sure this threshold is not too high. We don't want to set rough surfaces to zero.

            # define low lying (<mean) pixels inside valid area.
            # valid_flats=np.logical_and(valid_nanmask, grid.dir==0) #I thought grid.dir=0 meant flats. But this is not the case always apparently.
            valid_flats = np.logical_and(valid_nanmask, gMag < gMagTh)
            valid_low_flats = np.logical_and(valid_flats, grid.inflated_dem < mean_height)
            hand[valid_low_flats] = 0
        if np.any(np.isnan(hand)):
            grid.hand = fill_nan(hand)
    return hand


def get_land_mask(hand, dem_file, dem):
    # land mask
    nodata_fill_value = np.finfo(float).eps
    if np.any(np.isnan(hand)):
        print(f'{np.sum(np.isnan(hand))} NaN Pixels Detected in hand_result')
        # generate nan_mask
        # hand_type=hand.dtype
        # hand_orig=hand.copy()
        nan_mask = np.isnan(hand)
        # Download GSHHG
        hybas_dir = '/media/jzhu4/data/hand/external_data'  # if you do not want to keep any hybas files set it as hybas_dir=temp_dir
        gshhg_dir = '/media/jzhu4/data/hand/external_data'  # if you do not want to keep the coastline file, set it as gshhg_dir=temp_dir
        gshhg_url = 'http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip'
        gshhg_zipfile = os.path.join(gshhg_dir, "gshhg-shp-2.3.7.zip")
        gshhg_file = os.path.join(gshhg_dir, "GSHHS_shp/f/GSHHS_f_L1.shp")
        if not os.path.exists(gshhg_zipfile) and not os.path.exists(gshhg_file):
            # !wget -O {gshhg_zipfile} http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip
            urllib.request.urlretrieve(gshhg_url, gshhg_zipfile)
        if not os.path.exists(gshhg_file):
            # !unzip {gshhg_zipfile}
            with zipfile.ZipFile(gshhg_zipfile, 'r') as zip_ref:
                zip_ref.extractall(path=gshhg_dir)
        # if needed warp gshhg
        if dem.crs.to_string() != 'EPSG:4326':
            print("DEM and GSHHG projections differ.")
            # read extent shapes
            gshhg_df = gpd.read_file(gshhg_file)
            shapes = fiona_read_vectorfile(gshhg_file)
            dem_gT = gdal_get_geotransform(dem_file)
            dem_proj4 = get_projection(dem_file)
            dem = rasterio.open(dem_file)
            bb = gdal_bounding_box(dem_file)
            bb_poly = list(bb)
            bb_poly.append(bb[0])
            dem_poly = shapely.geometry.Polygon(bb_poly)
            dem_poly_wgs84 = transform_shape(dem_poly, s_srs=dem.crs.to_string())

            # find intersecting shapes
            polygons = intersect(shapes, dem_poly_wgs84)
            gshhg = []
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                for p in polygons:
                    pt = transform_polygon(p.exterior.coords, s_srs='epsg:4326', t_srs='epsg:' + str(dem.crs.to_epsg()))
                    gshhg.append(point_coordinates_to_geometry(pt.exterior.coords))
        else:
            gshhg = fiona_read_vectorfile(gshhg_file)
        # generate land_mask for the DEM

        land_mask, tf, win = rasterio.mask.raster_geometry_mask(dem, gshhg, crop=False,
                                                                invert=True)  # invert=If False (default), mask will be False inside shapes and True outside
        # set ocean/sea values in hand to epsilon
        if nodata_fill_value is not None:
            hand[np.invert(land_mask)] = nodata_fill_value  # sea_mask=np.invert(land_mask)
        # find nan areas that are within land_mask
        joint_mask = np.bitwise_and(nan_mask, land_mask)
        mask_labels, num_labels = ndimage.label(joint_mask)
        print(f"Number of NaN areas to fill: {num_labels}")
    # print_duration(t)
    return hand, mask_labels, num_labels, joint_mask


def fill_data_with_nan(hand, dem, mask_labels, num_labels, joint_mask):
    # new nan_fill needs DEM. Might be better to NOT load it in the memory See: https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
    demarray = dem.read(1)

    if np.any(np.isnan(hand)):
        object_slices = ndimage.find_objects(mask_labels)
        tq = tqdm(range(1, num_labels))
        for l in tq:  # Skip first, largest label.
            # ids = np.argwhere(mask_labels==l)
            # min0=max(ids[:,0].min()-1, 0)
            # max0=min(ids[:,0].max()+1, mask_labels.shape[0])
            # min1=max(ids[:,1].min()-1, 0)
            # max1=min(ids[:,1].max()+1, mask_labels.shape[1])
            slices = object_slices[l - 1]  # osl label=1 is in object_slices[0]
            min0 = max(slices[0].start - 1, 0)
            max0 = min(slices[0].stop + 1, mask_labels.shape[0])
            min1 = max(slices[1].start - 1, 0)
            max1 = min(slices[1].stop + 1, mask_labels.shape[1])
            mask_labels_clip = mask_labels[min0:max0, min1:max1]
            h = hand[min0:max0, min1:max1]  # by reference
            d = demarray[min0:max0, min1:max1]
            m = joint_mask[min0:max0, min1:max1].copy()
            m[mask_labels_clip != l] = 0  # Maskout other flooded areas (labels) for this area. Use only one label.
            if np.size(m) > 1e6:
                num_nan = m.sum()
                tq.set_description(f"Size: {num_nan}")
                if num_nan < 1e6:
                    # hf=fill_nan(h.copy()) #break reference
                    hf = fill_nan_based_on_dem(h.copy(), d.copy())
                    h[m] = hf[m]  # copy nanfill by reference
                else:
                    print(f'Filling {num_nan} pixels')
                    print('This can take a long time...')
                    hf = fill_nan_based_on_dem(h.copy(), d.copy())  # break reference
                    h[m] = hf[m]  # copy nanfill by reference
            else:
                hf = fill_nan_based_on_dem(h.copy(), d.copy())  # break reference
                h[m] = hf[m]  # copy nanfill by reference
    return hand


def calculate_hand_for_basins_orig(out_raster:  Union[str, Path], geometries: GeometryCollection,
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


def calculate_hand_for_basins(out_raster:  Union[str, Path], geometries: GeometryCollection,
                              dem_file: Union[str, Path]):
    nodata_fill_value = np.finfo(float).eps
    # dem_nodata_value = 0
    # Loop over each basin and calculate HAND
    # with rasterio.open(dem_file) as dem:
    dem = rasterio.open(dem_file, 'r')
    dem_nodata_value = dem.nodatavals[0]
    if dem_nodata_value is None:
        print('DEM does not have a defined no-data value.')
        print('Assuming all valid pixels. If not, expect long processing times.')
        dem_nodata_mask = np.zeros(dem.shape, dtype=np.byte)
    else:
        dem_nodata_mask = dem.read(1) == dem_nodata_value

    hand = np.zeros(dem.shape)
    hand[:] = np.nan  # set the hand to nan to make sure untouched pixels remain that value and not zero, which is a valid HAND height.
    for k, p in enumerate(tqdm(geometries.geoms)):
        verbose = False
        mask, tf, win = rasterio.mask.raster_geometry_mask(dem, [p], all_touched=True, crop=True, pad=True,
                                                           pad_width=1)  # add 1 pixel. calculate_hand needs it.
        # basin_mask, basin_affine_tf, basin_window = rasterio.mask.raster_geometry_mask(
        #    src, geometries, all_touched=True, crop=True, pad=True, pad_width=1)

        if win.width == 1 or win.height == 1:  # padding may require this limit to change.
            continue  # The DEM is a thin line, skip this patch
        not_mask = np.bitwise_not(mask)
        # if polygon_ids[k] == 4120928640: #k=15 for polygon_ids, in hybas_id[70883]
        #    verbose=True
        if dem_nodata_mask[win.row_off:win.row_off + win.height, win.col_off:win.col_off + win.width].all():
            continue  # do not process if the entire polygon is nodata
        # with warnings.catch_warnings():
        #    warnings.filterwarnings("ignore", category=FutureWarning)
            # h = calculate_hand(np.squeeze(src.read(window=win)), tf, pyproj.Proj(init=src.crs.to_string()),
            #                   mask=not_mask, verbose=verbose, acc_thresh=accumulation_threshold)
        if dem_nodata_mask[win.row_off:win.row_off + win.height, win.col_off:win.col_off + win.width].all():
            continue  # do not process if the entire polygon is nodata
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            h = calculate_hand(np.squeeze(dem.read(window=win)), tf, pyproj.Proj(init=dem.crs.to_string()),
                               mask=not_mask)
            # calculate_hand(dem_array, dem_gT, dem_proj4, mask=None, verbose=False, acc_thresh=100)
            # h = calculate_hand(dem_array, tf, proj, not_mask)
        clip_hand = hand[win.row_off:win.row_off + win.height, win.col_off:win.col_off + win.width]  # By reference
        clip_hand[not_mask] = h[not_mask]

    hand[dem_nodata_mask] = nodata_fill_value

    #write_cog(str(out_raster), hand, transform=tf.to_gdal(), epsg_code=src.crs.to_epsg())

    # create land mask
    hand, mask_labels, num_labels, joint_mask = get_land_mask(hand, dem_file, dem)
    # fill nan
    hand = fill_data_with_nan(hand, dem, mask_labels, num_labels, joint_mask)

    # write the hand
    # hand_file = os.path.splitext(dem_file)[0]+f"_hand_{version.replace('.','_')}.tif"
    dem_gT = gdal_get_geotransform(dem_file)
    dem_proj4 = get_projection(dem_file)
    gdal_write(hand, dem_gT, filename=str(out_raster), srs_proj4=dem_proj4, nodata=np.nan, data_format=gdal.GDT_Float32)


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
        # cut off the dem_vrt with envelop of geometries
        dem_file = get_tight_dem(dem_vrt.name, vector_file)
        calculate_hand_for_basins(out_raster, geometries, dem_file)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='For watershed boundaries, see: https://www.hydrosheds.org/page/hydrobasins',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('out_raster', type=Path,
                        help='HAND GeoTIFF to create')
    parser.add_argument('vector_file', type=Path,
                        help='Vector file of watershed boundary (hydrobasin) polygons to calculate HAND over. '
                             'Vector file Must be openable by GDAL, see: https://gdal.org/drivers/vector/index.html')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    log.debug(' '.join(sys.argv))
    log.info(f'Calculating HAND for {args.vector_file}')

    make_copernicus_hand(args.out_raster, args.vector_file)

    log.info(f'HAND GeoTIFF created successfully: {args.out_raster}')
