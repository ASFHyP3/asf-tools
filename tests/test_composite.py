import os

import numpy as np
import pytest
from osgeo import gdal

from asf_tools import composite


def test_get_epsg_code():
    wkt = 'PROJCS["WGS 84 / UTM zone 54N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",141],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32654"]]'
    info = {'coordinateSystem': {'wkt': wkt}}
    assert composite.get_epsg_code(info) == 32654

    wkt = 'PROJCS["WGS 84 / UTM zone 22N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-51],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32622"]]'
    info = {'coordinateSystem': {'wkt': wkt}}
    assert composite.get_epsg_code(info) == 32622

    wkt = 'PROJCS["WGS 84 / UTM zone 33S",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",15],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",10000000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32733"]]'
    info = {'coordinateSystem': {'wkt': wkt}}
    assert composite.get_epsg_code(info) == 32733

    wkt = 'PROJCS["NAD83 / Alaska Albers",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",50],PARAMETER["longitude_of_center",-154],PARAMETER["standard_parallel_1",55],PARAMETER["standard_parallel_2",65],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3338"]]'
    info = {'coordinateSystem': {'wkt': wkt}}
    assert composite.get_epsg_code(info) == 3338


def test_epsg_to_wkt():
    wkt = 'PROJCS["WGS 84 / UTM zone 54N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",141],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32654"]]'
    assert composite.epsg_to_wkt(32654) == wkt

    wkt = 'PROJCS["WGS 84 / UTM zone 22N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-51],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32622"]]'
    assert composite.epsg_to_wkt(32622) == wkt

    wkt = 'PROJCS["WGS 84 / UTM zone 33S",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",15],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",10000000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32733"]]'
    assert composite.epsg_to_wkt(32733) == wkt

    wkt = 'PROJCS["NAD83 / Alaska Albers",GEOGCS["NAD83",DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4269"]],PROJECTION["Albers_Conic_Equal_Area"],PARAMETER["latitude_of_center",50],PARAMETER["longitude_of_center",-154],PARAMETER["standard_parallel_1",55],PARAMETER["standard_parallel_2",65],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3338"]]'
    assert composite.epsg_to_wkt(3338) == wkt


def test_get_target_epsg_code():
    # Northern hemisphere
    assert composite.get_target_epsg_code([32601]) == 32601
    assert composite.get_target_epsg_code([32601, 32601]) == 32601

    # both hemispheres
    assert composite.get_target_epsg_code([32601, 32702]) == 32601
    assert composite.get_target_epsg_code([32702, 32601]) == 32601

    # Southern hemisphere
    assert composite.get_target_epsg_code([32760]) == 32760
    assert composite.get_target_epsg_code([32730, 32732]) == 32730

    # antimeridian
    assert composite.get_target_epsg_code([32701, 32760]) == 32701
    assert composite.get_target_epsg_code([32701, 32760, 32701]) == 32701
    assert composite.get_target_epsg_code([32701, 32760, 32760]) == 32760

    assert composite.get_target_epsg_code(
        [32731, 32631, 32731, 32631, 32732, 32633, 32733, 32633, 32733]
    ) == 32732

    # bounds
    with pytest.raises(ValueError):
        composite.get_target_epsg_code([32600])
    with pytest.raises(ValueError):
        composite.get_target_epsg_code([32661])
    with pytest.raises(ValueError):
        composite.get_target_epsg_code([32700])
    with pytest.raises(ValueError):
        composite.get_target_epsg_code([32761])
    with pytest.raises(ValueError):
        composite.get_target_epsg_code([32601, 99, 32760])


def test_get_area_raster():
    raster = 'S1A_IW_20181102T155531_DVP_RTC30_G_gpuned_5685_VV.tif'
    assert composite.get_area_raster(raster) == 'S1A_IW_20181102T155531_DVP_RTC30_G_gpuned_5685_area.tif'

    raster = './foo/S1B_IW_20181104T030247_DVP_RTC30_G_gpuned_9F91_VH.tif'
    assert composite.get_area_raster(raster) == './foo/S1B_IW_20181104T030247_DVP_RTC30_G_gpuned_9F91_area.tif'

    raster = '/tmp/bar/S1B_IW_20181102T031956_DVP_RTC30_G_gpuned_1259_HH.tif'
    assert composite.get_area_raster(raster) == '/tmp/bar/S1B_IW_20181102T031956_DVP_RTC30_G_gpuned_1259_area.tif'


def test_get_full_extents():
    data = {}

    data['a'] = {
        'cornerCoordinates': {
            'upperLeft': [10.0, 130.0],
            'lowerRight': [110.0, 30.0],
        },
        'geoTransform': [10.0, 2.0, 0.0, 40.0, 0.0, -2.0],
    }

    expected_upper_left = (10.0, 130.0)
    expected_lower_right = (110.0, 30.0)
    expected_geotransform = [10.0, 2.0, 0.0, 130.0, 0.0, -2.0]
    assert composite.get_full_extent(data) == (expected_upper_left, expected_lower_right, expected_geotransform)

    data['b'] = {
        'cornerCoordinates': {
            'upperLeft': [20.0, 140.0],
            'lowerRight': [120.0, 40.0],
        },
        'geoTransform': [20.0, 1.0, 12.0, 140.0, 13.0, -1.0],
    }

    expected_upper_left = (10.0, 140.0)
    expected_lower_right = (120.0, 30.0)
    expected_geotransform = [10.0, 2.0, 0.0, 140.0, 0.0, -2.0]
    assert composite.get_full_extent(data) == (expected_upper_left, expected_lower_right, expected_geotransform)


def test_write_cog(tmp_path):
    outfile = tmp_path / 'out.tif'
    data = np.ones((1024, 1024))
    transform = [10.0, 0.0, 1.0, 20.0, 0.0, -1.0]
    epsg_code = 4326

    result = composite.write_cog(str(outfile), data, transform, epsg_code)
    assert result == str(outfile)
    assert outfile.exists()

    info = gdal.Info(result, format='json')
    assert info['geoTransform'] == transform
    assert info['driverShortName'] == 'GTiff'
    assert info['size'] == [1024, 1024]
    assert 'overviews' in info['bands'][0]
    assert info['metadata']['IMAGE_STRUCTURE']['LAYOUT'] == 'COG'
    assert info['metadata']['IMAGE_STRUCTURE']['COMPRESSION'] == 'LZW'


def test_make_composite(tmp_path):
    os.chdir(tmp_path)
    epsg_code = 32601

    transform = [0.0, 30.0, 0.0, 60.0, 0.0, -30.0]
    data = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ])
    area = np.array([
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ])
    composite.write_cog('first_data.tif', data, transform, epsg_code, nodata_value=0)
    composite.write_cog('first_area.tif', area, transform, epsg_code)

    transform = [30.0, 30.0, 0.0, 30.0, 0.0, -30.0]
    data = np.array([
        [3, 0, 3, 3],
        [3, 0, 3, 3],
    ])
    area = np.array([
        [1, 1, 3, 1],
        [1, 1, 2, 1],
    ])
    composite.write_cog('second_data.tif', data, transform, epsg_code)
    composite.write_cog('second_area.tif', area, transform, epsg_code)

    out_file, count_file = composite.make_composite('out', ['first_data.tif', 'second_data.tif'])

    assert out_file == 'out.tif'
    assert count_file == 'out_counts.tif'
    assert os.path.exists(out_file)
    assert os.path.exists(count_file)

    data = np.nan_to_num(composite.read_as_array(out_file))
    expected = np.array([
        [1, 1, 1,   1, 0],
        [1, 2, 1, 1.5, 3],
        [0, 3, 0,   3, 3],
    ])
    assert np.allclose(data, expected)

    counts = composite.read_as_array(count_file)
    expected = np.array([
        [1, 1, 1, 1, 0],
        [1, 2, 1, 2, 1],
        [0, 1, 0, 1, 1],
    ])
    assert np.allclose(counts, expected)
