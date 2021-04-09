import pytest
from osgeo import gdal
from osgeo.utils.gdalcompare import find_diff

from asf_tools.water_map import make_water_map

gdal.UseExceptions()

@pytest.mark.integration
def test_initial_water_map(tmp_path, rtc_raster_pair, golden_water_map):
    primary, secondary = rtc_raster_pair

    out_water_map = tmp_path / 'initial_water_map.tif'
    make_water_map(out_water_map, primary, secondary)

    assert out_water_map.exists()

    diffs = find_diff(str(golden_water_map), str(out_water_map))
    assert diffs == 0
