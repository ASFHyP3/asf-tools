import numpy as np
import pytest
from osgeo.utils.gdalcompare import find_diff

from asf_tools import water_map
from asf_tools.composite import read_as_array
from asf_tools.util import tile_array


def test_determine_em_threshold(raster_tiles):
    scaling = 8.732284197109262
    threshold = water_map.determine_em_threshold(raster_tiles, scaling)
    assert np.isclose(threshold, 27.482176801248677)


@pytest.mark.integration
def test_select_hand_tiles(hand_candidates):
    hand = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/asf-tools/water-map/20200603_HAND.tif'
    hand_array = read_as_array(str(hand))
    hand_tiles = np.ma.masked_invalid(tile_array(hand_array, tile_shape=(100, 100), pad_value=np.nan))

    selected_tiles = water_map.select_hand_tiles(hand_tiles, 15., 0.8)
    assert np.all(selected_tiles == hand_candidates)


@pytest.mark.integration
def test_select_backscatter_tiles(hand_candidates):
    primary = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/asf-tools/water-map/20200603_VH.tif'
    primary_array = np.ma.masked_invalid(read_as_array(primary))
    primary_tiles = np.ma.masked_less_equal(tile_array(primary_array, tile_shape=(100, 100), pad_value=0.), 0.)

    selected_tiles = water_map.select_backscatter_tiles(primary_tiles, hand_candidates)
    assert np.all(selected_tiles == np.array([771, 1974, 2397, 1205, 2577]))


@pytest.mark.integration
def test_initial_water_map(tmp_path):
    primary = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/asf-tools/water-map/20200603_VH.tif'
    secondary = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/asf-tools/water-map/20200603_VV.tif'
    hand = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/asf-tools/water-map/20200603_HAND.tif'

    out_water_map = tmp_path / 'initial_water_map.tif'
    water_map.make_water_map(out_water_map, primary, secondary, hand)

    assert out_water_map.exists()

    golden_water_map = '/vsicurl/https://hyp3-testing.s3-us-west-2.amazonaws.com/asf-tools/water-map/' \
                       'em-threshold-initial-water-map.tif'
    diffs = find_diff(golden_water_map, str(out_water_map))
    assert diffs == 0
