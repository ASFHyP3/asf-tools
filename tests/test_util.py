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


def test_tile_masked_array():
    a = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [2, 2, 3, 3],
                  [2, 2, 3, 3]])

    with pytest.raises(AttributeError):
        _ = util.tile_array(a, tile_shape=(2, 2)).mask

    m = np.array([[False, False, False, True],
                  [False, False, False, False],
                  [False, False, False, False],
                  [False, False, False, True]])

    ma = np.ma.MaskedArray(a, mask=m)
    tiled = util.tile_array(ma, tile_shape=(2, 2))

    assert tiled.shape == (4, 2, 2)
    assert isinstance(tiled, np.ma.MaskedArray)
    assert np.all(
        tiled.mask == np.array([[[False, False],
                                 [False, False]],
                                [[False, True],
                                 [False, False]],
                                [[False, False],
                                 [False, False]],
                                [[False, False],
                                 [False, True]]])
    )


def test_untile_array():
    a = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [2, 2, 3, 3],
                  [2, 2, 3, 3]])

    assert np.all(a == util.untile_array(util.tile_array(a, tile_shape=(2, 2)), array_shape=a.shape))
    assert np.all(a == util.untile_array(util.tile_array(a, tile_shape=(3, 3), pad_value=4), array_shape=a.shape))
    assert np.all(a == util.untile_array(util.tile_array(a, tile_shape=(2, 3), pad_value=4), array_shape=a.shape))
    assert np.all(a == util.untile_array(util.tile_array(a, tile_shape=(3, 2), pad_value=4), array_shape=a.shape))

    with pytest.raises(ValueError):
        util.untile_array(util.tile_array(a, tile_shape=(2, 2)), array_shape=(5, 5))

    with pytest.raises(ValueError):
        util.untile_array(util.tile_array(a, tile_shape=(2, 3), pad_value=4), array_shape=(4, 7))

    # array shape will subset some of the padding that was required to tile `a` with `tile_shape`
    assert np.all(
            np.pad(a, ((0, 0), (0, 1)), constant_values=4)
            == util.untile_array(util.tile_array(a, tile_shape=(2, 3), pad_value=4), array_shape=(4, 5))
    )


def test_untile_masked_array():
    a = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [2, 2, 3, 3],
                  [2, 2, 3, 3]])

    with pytest.raises(AttributeError):
        _ = util.untile_array(util.tile_array(a, tile_shape=(2, 2)), array_shape=a.shape).mask

    m = np.array([[False, False, False, True],
                  [False, False, False, False],
                  [False, False, False, False],
                  [False, False, False, True]])

    ma = np.ma.MaskedArray(a, mask=m)
    untiled = util.untile_array(util.tile_array(ma.copy(), tile_shape=(2, 2)), array_shape=a.shape)

    assert np.all(ma == untiled)
    assert np.all(ma.mask == untiled.mask)
