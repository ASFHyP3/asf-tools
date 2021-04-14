from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator, List, Union

import shapely.geometry
from osgeo import gdal, ogr

__all__ = ['GDALConfigManager', 'prepare_dem_vrt']
DEM_GEOJSON = '/vsicurl/https://asf-dem-west.s3.amazonaws.com/v2/cop30.geojson'

gdal.UseExceptions()
ogr.UseExceptions()


class GDALConfigManager:
    """Context manager for setting GDAL config options temporarily"""
    def __init__(self, **options):
        """
        Args:
            **options: GDAL Config `option=value` keyword arguments.
        """
        self.options = options.copy()
        self._previous_options = {}

    def __enter__(self):
        for key in self.options:
            self._previous_options[key] = gdal.GetConfigOption(key)

        for key, value in self.options.items():
            gdal.SetConfigOption(key, value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self._previous_options.items():
            gdal.SetConfigOption(key, value)


def get_dem_features() -> Generator[ogr.Feature, None, None]:
    ds = ogr.Open(DEM_GEOJSON)
    layer = ds.GetLayer()
    for feature in layer:
        yield feature
    del ds


def intersects_dem(geometry: ogr.Geometry) -> bool:
    for feature in get_dem_features():
        if feature.GetGeometryRef().Intersects(geometry):
            return True


def get_dem_file_paths(geometry: ogr.Geometry) -> List[str]:
    file_paths = []
    for feature in get_dem_features():
        if feature.GetGeometryRef().Intersects(geometry):
            file_paths.append(feature.GetField('file_path'))
    return file_paths


def shift_for_antimeridian(dem_file_paths: List[str], directory: Path) -> List[str]:
    shifted_file_paths = []
    for file_path in dem_file_paths:
        if '_W' in file_path:
            shifted_file_path = str(directory / Path(file_path).with_suffix('.vrt').name)
            corners = gdal.Info(file_path, format='json')['cornerCoordinates']
            output_bounds = [
                corners['upperLeft'][0] + 360,
                corners['upperLeft'][1],
                corners['lowerRight'][0] + 360,
                corners['lowerRight'][1]
            ]
            gdal.Translate(shifted_file_path, file_path, format='VRT', outputBounds=output_bounds)
            shifted_file_paths.append(shifted_file_path)
        else:
            shifted_file_paths.append(file_path)
    return shifted_file_paths


def prepare_dem_vrt(vrt: Union[str, Path], geometry: Union[ogr.Geometry, shapely.geometry.GeometryCollection]):
    """Create a DEM mosaic VRT covering a given geometry

    The DEM mosaic is assembled from the Copernicus GLO-30 Public DEM tiles that intersect the geometry.

    Note: If the input geometry is a MULTIPOLYGON, this assumes the polygons are adjacent to the antimeridian.

    Args:
        vrt: Path for the output VRT file
        geometry: Geometry in EPSG:4326 (lon/lat) projection for which to prepare a DEM mosaic

    """
    with GDALConfigManager(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
        if isinstance(geometry, shapely.geometry.GeometryCollection):
            geometry = ogr.CreateGeometryFromWkb(geometry.wkb)

        if not intersects_dem(geometry):
            raise ValueError(f'Copernicus GLO-30 Public DEM does not intersect this geometry: {geometry}')

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dem_file_paths = get_dem_file_paths(geometry)

            if geometry.GetGeometryName() == 'MULTIPOLYGON':
                dem_file_paths = shift_for_antimeridian(dem_file_paths, temp_path)

            gdal.BuildVRT(str(vrt), dem_file_paths)
