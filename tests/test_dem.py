import json

import pytest
from osgeo import gdal, ogr

from asf_tools import dem

gdal.UseExceptions()


def test_intersects_dem():
    geojson = {
        'type': 'Point',
        'coordinates': [169, -45],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    assert dem.intersects_dem(geometry)

    geojson = {
        'type': 'Point',
        'coordinates': [0, 0],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    assert not dem.intersects_dem(geometry)


def test_get_file_paths():
    geojson = {
        'type': 'Point',
        'coordinates': [0, 0],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    assert dem.get_dem_file_paths(geometry) == []

    geojson = {
        'type': 'Point',
        'coordinates': [169, -45],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    assert dem.get_dem_file_paths(geometry) == [
        '/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/'
        'Copernicus_DSM_COG_10_S46_00_E169_00_DEM/Copernicus_DSM_COG_10_S46_00_E169_00_DEM.tif'
    ]

    geojson = {
        'type': 'MultiPoint',
        'coordinates': [[0, 0], [169, -45], [-121.5, 73.5]]
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    assert dem.get_dem_file_paths(geometry) == [
        '/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/'
        'Copernicus_DSM_COG_10_S46_00_E169_00_DEM/Copernicus_DSM_COG_10_S46_00_E169_00_DEM.tif',
        '/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/'
        'Copernicus_DSM_COG_10_N73_00_W122_00_DEM/Copernicus_DSM_COG_10_N73_00_W122_00_DEM.tif',
    ]


def test_get_dem_features():
    assert len(list(dem.get_dem_features())) == 26445


def test_shift_for_antimeridian(tmp_path):
    file_paths = [
        '/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/'
        'Copernicus_DSM_COG_10_N51_00_W180_00_DEM/Copernicus_DSM_COG_10_N51_00_W180_00_DEM.tif',
        '/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/'
        'Copernicus_DSM_COG_10_N51_00_E179_00_DEM/Copernicus_DSM_COG_10_N51_00_E179_00_DEM.tif'
    ]

    with dem.GDALConfigManager(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
        shifted_file_paths = dem.shift_for_antimeridian(file_paths, tmp_path)

    assert shifted_file_paths[0] == str(tmp_path / 'Copernicus_DSM_COG_10_N51_00_W180_00_DEM.vrt')
    assert shifted_file_paths[1] == file_paths[1]

    info = gdal.Info(shifted_file_paths[0], format='json')
    assert info['cornerCoordinates']['upperLeft'] == [179.9997917, 52.0001389]
    assert info['cornerCoordinates']['lowerRight'] == [180.9997917, 51.0001389]


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

    dem.prepare_dem_vrt(str(dem_vrt), geometry)
    assert dem_vrt.exists()

    info = gdal.Info(str(dem_vrt), format='json')
    assert info['geoTransform'] == \
           [178.99979166666665, 0.0004166666666667, 0.0, 52.0001389, 0.0, -0.0002777777777778]
    assert info['size'] == [4800, 3600]
