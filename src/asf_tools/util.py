from typing import Tuple

from osgeo import gdal, osr

gdal.UseExceptions()


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


def get_epsg_code(info: dict) -> int:
    """Get the EPSG code from a GDAL Info dictionary

    Args:
        info: The dictionary returned by a gdal.Info call

    Returns:
        epsg_code: The integer EPSG code
    """
    proj = osr.SpatialReference(info['coordinateSystem']['wkt'])
    epsg_code = int(proj.GetAttrValue('AUTHORITY', 1))
    return epsg_code


def get_coordinates(info: dict) -> Tuple[int, int, int, int]:
    """Get the corner coordinates from a GDAL Info dictionary

    Args:
        info: The dictionary returned by a gdal.Info call

    Returns:
        (west, south, east, north): the corner coordinates values
    """
    west, south = info['cornerCoordinates']['lowerLeft']
    east, north = info['cornerCoordinates']['upperRight']
    return west, south, east, north


def epsg_to_wkt(epsg_code: int) -> str:
    """Get the WKT representation of a projection from its EPSG code

    Args:
        epsg_code: The integer EPSG code

    Returns:
        wkt: The WKT representation of the projection
    """
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    return srs.ExportToWkt()
