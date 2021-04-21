import numpy as np
import pytest
from osgeo.utils.gdalcompare import find_diff

from asf_tools import water_map


def test_determine_em_threshold(raster_tiles):
    scaling = 8.732284197109262
    threshold = water_map.determine_em_threshold(raster_tiles, scaling)
    assert np.isclose(threshold, 27.482176801248677)


@pytest.mark.integration
def test_initial_water_map(tmp_path, rtc_raster_pair, golden_water_map):
    primary, secondary = rtc_raster_pair

    out_water_map = tmp_path / 'initial_water_map.tif'
    water_map.make_water_map(out_water_map, primary, secondary)

    assert out_water_map.exists()

    diffs = find_diff(str(golden_water_map), str(out_water_map))
    assert diffs == 0
