"""Prepare a Height Above Nearest Drainage (HAND) virtual raster (VRT) covering a given geometry"""
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator, List, Union

import shapely.geometry
from osgeo import gdal, ogr

from asf_tools.dem import GDALConfigManager, shift_for_antimeridian

HAND_GEOJSON = '/vsicurl/https://asf-hand-data.s3.amazonaws.com/cop30-hand.geojson'

gdal.UseExceptions()
ogr.UseExceptions()


def get_hand_features() -> Generator[ogr.Feature, None, None]:
    ds = ogr.Open(HAND_GEOJSON)
    layer = ds.GetLayer()
    for feature in layer:
        yield feature
    del ds


def intersects_hand(geometry: ogr.Geometry) -> bool:
    for feature in get_hand_features():
        if feature.GetGeometryRef().Intersects(geometry):
            return True


def get_hand_file_paths(geometry: ogr.Geometry) -> List[str]:
    file_paths = []
    for feature in get_hand_features():
        if feature.GetGeometryRef().Intersects(geometry):
            file_paths.append(feature.GetField('file_path'))
    return file_paths


def prepare_hand_vrt(vrt: Union[str, Path], geometry: Union[ogr.Geometry, shapely.geometry.GeometryCollection]):
    """Prepare a HAND mosaic VRT covering a given geometry

    Prepare a Height Above Nearest Drainage (HAND) virtual raster (VRT) covering a given geometry.
    The Height Above Nearest Drainage (HAND) mosaic is assembled from the HAND tiles that intersect
    the geometry, using a HAND derived from the Copernicus GLO-30 DEM.

    Note: If the input geometry is a MULTIPOLYGON, this assumes the polygons are adjacent to the antimeridian.

    Args:
        vrt: Path for the output VRT file
        geometry: Geometry in EPSG:4326 (lon/lat) projection for which to prepare a DEM mosaic

    """
    with GDALConfigManager(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
        if isinstance(geometry, shapely.geometry.GeometryCollection):
            geometry = ogr.CreateGeometryFromWkb(geometry.wkb)

        if not intersects_hand(geometry):
            raise ValueError(f'Copernicus GLO-30 HAND does not intersect this geometry: {geometry}')

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dem_file_paths = get_hand_file_paths(geometry)

            if geometry.GetGeometryName() == 'MULTIPOLYGON':
                dem_file_paths = shift_for_antimeridian(dem_file_paths, temp_path)

            gdal.BuildVRT(str(vrt), dem_file_paths)
