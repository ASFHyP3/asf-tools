import os

import numpy as np
import pytest

import asf_tools.raster
from asf_tools import composite


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
    asf_tools.raster.write_cog('first_data.tif', data, transform, epsg_code, nodata_value=0)
    asf_tools.raster.write_cog('first_area.tif', area, transform, epsg_code)

    transform = [30.0, 30.0, 0.0, 30.0, 0.0, -30.0]
    data = np.array([
        [3, 0, 3, 3],
        [3, 0, 3, 3],
    ])
    area = np.array([
        [1, 1, 3, 1],
        [1, 1, 2, 1],
    ])
    asf_tools.raster.write_cog('second_data.tif', data, transform, epsg_code)
    asf_tools.raster.write_cog('second_area.tif', area, transform, epsg_code)

    out_file, count_file = composite.make_composite('out', ['first_data.tif', 'second_data.tif'])

    assert out_file == 'out.tif'
    assert count_file == 'out_counts.tif'
    assert os.path.exists(out_file)
    assert os.path.exists(count_file)

    data = np.nan_to_num(asf_tools.raster.read_as_array(out_file))
    expected = np.array([
        [1, 1, 1,   1, 0],
        [1, 2, 1, 1.5, 3],
        [0, 3, 0,   3, 3],
    ])
    assert np.allclose(data, expected)

    counts = asf_tools.raster.read_as_array(count_file)
    expected = np.array([
        [1, 1, 1, 1, 0],
        [1, 2, 1, 2, 1],
        [0, 1, 0, 1, 1],
    ])
    assert np.allclose(counts, expected)
