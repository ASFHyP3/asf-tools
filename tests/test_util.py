import numpy as np
import pytest

from asf_tools import util


def test_tile_array():
    a = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [2, 2, 3, 3],
                  [2, 2, 3, 3]])

    tiled = util.tile_array(a, tile_shape=(2, 2))
    assert tiled.shape == (4, 2, 2)
    assert np.all(tiled[0, :, :] == np.array([[0, 0], [0, 0]]))
    assert np.all(tiled[-1, :, :] == np.array([[3, 3], [3, 3]]))

    with pytest.raises(ValueError):
        util.tile_array(a, tile_shape=(3, 3))

    tiled = util.tile_array(a, tile_shape=(3, 3), pad_value=4)
    assert tiled.shape == (4, 3, 3)
    assert np.all(tiled[0, :, :] == np.array([[0, 0, 1], [0, 0, 1], [2, 2, 3]]))
    assert np.all(tiled[-1, :, :] == np.array([[3, 4, 4], [4, 4, 4], [4, 4, 4]]))

    tiled = util.tile_array(a, tile_shape=(2, 3), pad_value=4)
    assert tiled.shape == (4, 2, 3)
    assert np.all(tiled[0, :, :] == np.array([[0, 0, 1], [0, 0, 1]]))
    assert np.all(tiled[-1, :, :] == np.array([[3, 4, 4], [3, 4, 4]]))

    tiled = util.tile_array(a, tile_shape=(3, 2), pad_value=4)
    assert tiled.shape == (4, 3, 2)
    assert np.all(tiled[0, :, :] == np.array([[0, 0], [0, 0], [2, 2]]))
    assert np.all(tiled[-1, :, :] == np.array([[3, 3], [4, 4], [4, 4]]))


def test_untile_array():
    a = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [2, 2, 3, 3],
                  [2, 2, 3, 3]])

    assert np.all(a == util.untile_array(util.tile_array(a, tile_shape=(2, 2)), array_shape=a.shape))
    assert np.all(a == util.untile_array(util.tile_array(a, tile_shape=(3, 3), pad_value=4), array_shape=a.shape))
    assert np.all(a == util.untile_array(util.tile_array(a, tile_shape=(2, 3), pad_value=4), array_shape=a.shape))
    assert np.all(a == util.untile_array(util.tile_array(a, tile_shape=(3, 2), pad_value=4), array_shape=a.shape))
