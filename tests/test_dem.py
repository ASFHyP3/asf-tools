import json

import pytest
from osgeo import gdal, ogr

from asf_tools import dem

gdal.UseExceptions()


def test_prepare_dem_vrt_no_coverage():
    geojson = {
        'type': 'Point',
        'coordinates': [0, 0],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    with pytest.raises(ValueError):
        dem.prepare_dem_vrt('foo', geometry)


def test_prepare_dem_vrt(tmp_path):
    dem_vrt = tmp_path / 'dem.tif'
    geojson = {
        'type': 'Polygon',
        'coordinates': [[
            [0.4, 10.16],
            [0.4, 10.86],
            [0.6, 10.86],
            [0.6, 10.16],
            [0.4, 10.16],
        ]],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))

    dem.prepare_dem_vrt(str(dem_vrt), geometry)
    assert dem_vrt.exists()

    info = gdal.Info(str(dem_vrt), format='json')
    assert info['geoTransform'] == \
           [-0.0001388888888889, 0.0002777777777778, 0.0, 11.00013888888889, 0.0, -0.0002777777777778]
    assert info['size'] == [3600, 3600]


def test_prepare_dem_geotiff_antimeridian(tmp_path):
    dem_vrt = tmp_path / 'dem.vrt'
    geojson = {
        'type': 'MultiPolygon',
        'coordinates': [
            [[
                [179.5, 51.4],
                [179.5, 51.6],
                [180.0, 51.6],
                [180.0, 51.4],
                [179.5, 51.4],
            ]],
            [[
                [-180.0, 51.4],
                [-180.0, 51.6],
                [-179.5, 51.6],
                [-179.5, 51.4],
                [-180.0, 51.4],
            ]],
        ],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))

    with pytest.raises(ValueError):
        dem.prepare_dem_vrt(str(dem_vrt), geometry)
