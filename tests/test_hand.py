import json

import pytest
from osgeo import gdal, ogr
from osgeo.utils.gdalcompare import find_diff

from asf_tools import dem
from asf_tools import hand
from asf_tools.hand import prepare

HAND_BASINS = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/' \
              'asf-tools/hand/hybas_af_lev12_v1c_firstpoly.geojson'
GOLDEN_HAND = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/' \
              'asf-tools/hand/hybas_af_lev12_v1c_firstpoly.tif'

gdal.UseExceptions()


@pytest.mark.integration
def test_make_copernicus_hand(tmp_path):

    out_hand = tmp_path / 'hand.tif'
    hand.make_copernicus_hand(out_hand, HAND_BASINS)

    assert out_hand.exists()

    diffs = find_diff(str(GOLDEN_HAND), str(out_hand))
    assert diffs == 0


def test_intersects_hand():
    geojson = {
        'type': 'Point',
        'coordinates': [169, -45],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    assert prepare.intersects_hand(geometry)

    geojson = {
        'type': 'Point',
        'coordinates': [0, 0],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    assert not prepare.intersects_hand(geometry)


def test_get_file_paths():
    geojson = {
        'type': 'Point',
        'coordinates': [0, 0],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    assert prepare.get_hand_file_paths(geometry) == []

    geojson = {
        'type': 'Point',
        'coordinates': [169, -45],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    assert prepare.get_hand_file_paths(geometry) == [
        '/vsicurl/https://asf-hand-data.s3-us-west-2.amazonaws.com/'
        'GLOBAL_HAND/Copernicus_DSM_COG_10_S46_00_E169_00_HAND.tif'
    ]

    geojson = {
        'type': 'MultiPoint',
        'coordinates': [[0, 0], [169, -45], [-121.5, 73.5]]
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    assert prepare.get_hand_file_paths(geometry) == [
        '/vsicurl/https://asf-hand-data.s3-us-west-2.amazonaws.com/'
        'GLOBAL_HAND/Copernicus_DSM_COG_10_S46_00_E169_00_HAND.tif',
        '/vsicurl/https://asf-hand-data.s3-us-west-2.amazonaws.com/'
        'GLOBAL_HAND/Copernicus_DSM_COG_10_N73_00_W122_00_HAND.tif',
    ]


def test_get_hand_features():
    assert len(list(prepare.get_hand_features())) == 26445


def test_shift_for_antimeridian(tmp_path):
    file_paths = [
        '/vsicurl/https://asf-hand-data.s3-us-west-2.amazonaws.com/'
        'GLOBAL_HAND/Copernicus_DSM_COG_10_N51_00_W180_00_HAND.tif',
        '/vsicurl/https://asf-hand-data.s3-us-west-2.amazonaws.com/'
        'GLOBAL_HAND/Copernicus_DSM_COG_10_N51_00_E179_00_HAND.tif'
    ]

    with dem.GDALConfigManager(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
        shifted_file_paths = dem.shift_for_antimeridian(file_paths, tmp_path)

    assert shifted_file_paths[0] == str(tmp_path / 'Copernicus_DSM_COG_10_N51_00_W180_00_HAND.vrt')
    assert shifted_file_paths[1] == file_paths[1]

    info = gdal.Info(shifted_file_paths[0], format='json')
    assert info['cornerCoordinates']['upperLeft'] == [179.9997917, 52.0001389]
    assert info['cornerCoordinates']['lowerRight'] == [180.9997917, 51.0001389]


def test_prepare_hand_vrt_no_coverage():
    geojson = {
        'type': 'Point',
        'coordinates': [0, 0],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    with pytest.raises(ValueError):
        hand.prepare_hand_vrt('foo', geometry)


def test_prepare_hand_vrt(tmp_path):
    hand_vrt = tmp_path / 'hand.tif'
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

    hand.prepare_hand_vrt(str(hand_vrt), geometry)
    assert hand_vrt.exists()

    info = gdal.Info(str(hand_vrt), format='json')
    assert info['geoTransform'] == \
           [-0.0001388888888889, 0.0002777777777778, 0.0, 11.00013888888889, 0.0, -0.0002777777777778]
    assert info['size'] == [3600, 3600]


def test_prepare_hand_geotiff_antimeridian(tmp_path):
    hand_vrt = tmp_path / 'hand.vrt'
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

    hand.prepare_hand_vrt(str(hand_vrt), geometry)
    assert hand_vrt.exists()

    info = gdal.Info(str(hand_vrt), format='json')
    assert info['geoTransform'] == \
           [178.99979166666665, 0.0004166666666667, 0.0, 52.0001389, 0.0, -0.0002777777777778]
    assert info['size'] == [4800, 3600]
