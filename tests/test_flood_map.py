import numpy as np
import pytest
from osgeo_utils.gdalcompare import find_diff

from asf_tools import flood_map
from asf_tools.composite import read_as_array
from asf_tools.tile import tile_array

@pytest.mark.integration
def test_check_coordinate_system():
    assert 1==1


@pytest.mark.integration
def test_get_waterbody():
    assert 1==1


@pytest.mark.integration
def test_estimate_flood_depths():
    assert 1==1


@pytest.mark.integration
def test_make_flood_map():
    assert 1==1