"""Prepare a Copernicus GLO-30 DEM virtual raster (VRT) covering a given geometry"""
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import shapely.geometry
from osgeo import gdal, ogr

from asf_tools import raster
from asf_tools import vector
from asf_tools.util import GDALConfigManager

DEM_GEOJSON = '/vsicurl/https://asf-dem-west.s3.amazonaws.com/v2/cop30.geojson'

gdal.UseExceptions()
ogr.UseExceptions()


def prepare_dem_vrt(vrt: Union[str, Path], geometry: Union[ogr.Geometry, shapely.geometry.GeometryCollection]):
    """Create a DEM mosaic VRT covering a given geometry

    The DEM mosaic is assembled from the Copernicus GLO-30 DEM tiles that intersect the geometry.

    Note: If the input geometry is a MULTIPOLYGON, this assumes the polygons are adjacent to the antimeridian.

    Args:
        vrt: Path for the output VRT file
        geometry: Geometry in EPSG:4326 (lon/lat) projection for which to prepare a DEM mosaic

    """
    with GDALConfigManager(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
        if isinstance(geometry, shapely.geometry.GeometryCollection):
            geometry = ogr.CreateGeometryFromWkb(geometry.wkb)

        min_lon, max_lon, _, _ = geometry.GetEnvelope()
        if min_lon < -160. and max_lon > 160.:
            raise ValueError(f'ASF Tools does not currently support geometries that cross the antimeridian: {geometry}')

        tile_features = vector.get_features(DEM_GEOJSON)
        if not vector.intersects_features(geometry, tile_features):
            raise ValueError(f'Copernicus GLO-30 DEM does not intersect this geometry: {geometry}')

        tile_features = vector.get_features(DEM_GEOJSON)
        dem_file_paths = vector.intersecting_feature_properties(geometry, tile_features, 'file_path')

        gdal.BuildVRT(str(vrt), dem_file_paths)
