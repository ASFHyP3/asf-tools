import json

from osgeo import ogr

from asf_tools import dem, hand, vector


def test_intersects_feature():
    for vector_file in (dem.DEM_GEOJSON, hand.prepare.HAND_GEOJSON):
        features = vector.get_features(vector_file)

        geojson = {
            'type': 'Point',
            'coordinates': [169, -45],
        }
        geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
        assert vector.get_property_values_for_intersecting_features(geometry, features)

        geojson = {
            'type': 'Point',
            'coordinates': [0, 0],
        }
        geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
        assert not vector.get_property_values_for_intersecting_features(geometry, features)


def test_get_intersecting_feature_properties():
    dem_tile_features = vector.get_features(dem.DEM_GEOJSON)

    geojson = {
        'type': 'Point',
        'coordinates': [0, 0],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    assert vector.intersecting_feature_properties(geometry, dem_tile_features, 'file_path') == []

    geojson = {
        'type': 'Point',
        'coordinates': [169, -45],
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    assert vector.intersecting_feature_properties(geometry, dem_tile_features, 'file_path') == [
        '/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/'
        'Copernicus_DSM_COG_10_S46_00_E169_00_DEM/Copernicus_DSM_COG_10_S46_00_E169_00_DEM.tif'
    ]

    geojson = {
        'type': 'MultiPoint',
        'coordinates': [[0, 0], [169, -45], [-121.5, 73.5]]
    }
    geometry = ogr.CreateGeometryFromJson(json.dumps(geojson))
    assert vector.intersecting_feature_properties(geometry, dem_tile_features, 'file_path') == [
        '/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/'
        'Copernicus_DSM_COG_10_S46_00_E169_00_DEM/Copernicus_DSM_COG_10_S46_00_E169_00_DEM.tif',
        '/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/'
        'Copernicus_DSM_COG_10_N73_00_W122_00_DEM/Copernicus_DSM_COG_10_N73_00_W122_00_DEM.tif',
    ]


def test_get_features():
    for vector_file in (dem.DEM_GEOJSON, hand.prepare.HAND_GEOJSON):
        assert len(vector.get_features(vector_file)) == 26445
