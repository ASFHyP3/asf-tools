import pytest
from osgeo.utils.gdalcompare import find_diff

from asf_tools import hand


@pytest.mark.integration
def test_make_copernicus_hand(tmp_path, hand_basin, golden_hand):

    out_hand = tmp_path / 'hand.tif'
    hand.make_copernicus_hand(out_hand, hand_basin)

    assert out_hand.exists()

    diffs = find_diff(str(golden_hand), str(out_hand))
    assert diffs == 0
