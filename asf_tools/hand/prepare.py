"""Prepare a Height Above Nearest Drainage (HAND) virtual raster (VRT) covering a given geometry"""
from pathlib import Path
from typing import Union

import shapely.geometry
from osgeo import gdal, ogr

from asf_tools import vector
from asf_tools.util import GDALConfigManager

HAND_GEOJSON = '/vsicurl/https://asf-hand-data.s3.amazonaws.com/cop30-hand.geojson'

gdal.UseExceptions()
ogr.UseExceptions()


def prepare_hand_vrt(vrt: Union[str, Path], geometry: Union[ogr.Geometry, shapely.geometry.GeometryCollection]):
    """Prepare a HAND mosaic VRT covering a given geometry

    Prepare a Height Above Nearest Drainage (HAND) virtual raster (VRT) covering a given geometry.
    The Height Above Nearest Drainage (HAND) mosaic is assembled from the HAND tiles that intersect
    the geometry, using a HAND derived from the Copernicus GLO-30 DEM.

    Note: `asf_tools` does not currently support geometries that cross the antimeridian.

    Args:
        vrt: Path for the output VRT file
        geometry: Geometry in EPSG:4326 (lon/lat) projection for which to prepare a DEM mosaic

    """
    with GDALConfigManager(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
        if isinstance(geometry, shapely.geometry.GeometryCollection):
            geometry = ogr.CreateGeometryFromWkb(geometry.wkb)

        min_lon, max_lon, _, _ = geometry.GetEnvelope()
        if min_lon < -160. and max_lon > 160.:
            raise ValueError(f'asf_tools does not currently support geometries that cross the antimeridian: {geometry}')

        tile_features = vector.get_features(HAND_GEOJSON)
        if not vector.intersects_features(geometry, tile_features):
            raise ValueError(f'Copernicus GLO-30 HAND does not intersect this geometry: {geometry}')

        hand_file_paths = vector.intersecting_feature_properties(geometry, tile_features, 'file_path')

        gdal.BuildVRT(str(vrt), hand_file_paths)
