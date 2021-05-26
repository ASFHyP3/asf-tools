import numpy as np
import pytest

from asf_tools import tile


def test_tile_array():
    a = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [2, 2, 3, 3],
                  [2, 2, 3, 3]])

    tiled = tile.tile_array(a, tile_shape=(2, 2))
    assert tiled.shape == (4, 2, 2)
    assert np.all(tiled[0, :, :] == np.array([[0, 0], [0, 0]]))
    assert np.all(tiled[-1, :, :] == np.array([[3, 3], [3, 3]]))

    with pytest.raises(ValueError):
        tile.tile_array(a, tile_shape=(3, 3))

    tiled = tile.tile_array(a, tile_shape=(3, 3), pad_value=4)
    assert tiled.shape == (4, 3, 3)
    assert np.all(tiled[0, :, :] == np.array([[0, 0, 1], [0, 0, 1], [2, 2, 3]]))
    assert np.all(tiled[-1, :, :] == np.array([[3, 4, 4], [4, 4, 4], [4, 4, 4]]))

    tiled = tile.tile_array(a, tile_shape=(2, 3), pad_value=4)
    assert tiled.shape == (4, 2, 3)
    assert np.all(tiled[0, :, :] == np.array([[0, 0, 1], [0, 0, 1]]))
    assert np.all(tiled[-1, :, :] == np.array([[3, 4, 4], [3, 4, 4]]))

    tiled = tile.tile_array(a, tile_shape=(3, 2), pad_value=4)
    assert tiled.shape == (4, 3, 2)
    assert np.all(tiled[0, :, :] == np.array([[0, 0], [0, 0], [2, 2]]))
    assert np.all(tiled[-1, :, :] == np.array([[3, 3], [4, 4], [4, 4]]))


def test_tile_masked_array():
    a = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [2, 2, 3, 3],
                  [2, 2, 3, 3]])

    with pytest.raises(AttributeError):
        _ = tile.tile_array(a, tile_shape=(2, 2)).mask

    m = np.array([[False, False, False, True],
                  [False, False, False, False],
                  [False, False, False, False],
                  [False, False, False, True]])

    ma = np.ma.MaskedArray(a, mask=m)
    tiled = tile.tile_array(ma, tile_shape=(2, 2))

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

    tiled = tile.tile_array(ma, tile_shape=(3, 3), pad_value=4)
    assert isinstance(tiled, np.ma.MaskedArray)
    assert tiled.shape == (4, 3, 3)
    assert np.all(np.ma.getdata(tiled[0, :, :]) == np.array([[0, 0, 1], [0, 0, 1], [2, 2, 3]]))
    assert np.all(
        tiled[0, :, :].mask == np.array([[False, False, False], [False, False, False], [False, False, False]])
    )
    assert np.all(np.ma.getdata(tiled[-1, :, :]) == np.array([[3, 4, 4], [4, 4, 4], [4, 4, 4]]))
    assert np.all(
        tiled[-1, :, :].mask == np.array([[True, True, True], [True, True, True], [True, True, True]])
    )


def test_untile_array():
    a = np.array([[0, 0, 1, 1, 2, 2],
                  [0, 0, 1, 1, 2, 2],
                  [3, 3, 4, 4, 5, 5],
                  [3, 3, 4, 4, 5, 5],
                  [6, 6, 7, 7, 8, 8],
                  [6, 6, 7, 7, 8, 8],
                  ])

    assert np.all(a == tile.untile_array(tile.tile_array(a, tile_shape=(2, 2)), array_shape=a.shape))
    assert np.all(a == tile.untile_array(tile.tile_array(a, tile_shape=(4, 4), pad_value=9), array_shape=a.shape))
    assert np.all(a == tile.untile_array(tile.tile_array(a, tile_shape=(2, 4), pad_value=9), array_shape=a.shape))
    assert np.all(a == tile.untile_array(tile.tile_array(a, tile_shape=(4, 2), pad_value=9), array_shape=a.shape))

    with pytest.raises(ValueError):
        tile.untile_array(tile.tile_array(a, tile_shape=(4, 4)), array_shape=(9, 9))

    with pytest.raises(ValueError):
        tile.untile_array(tile.tile_array(a, tile_shape=(2, 4), pad_value=9), array_shape=(6, 9))

    # array shape will subset some of the padding that was required to tile `a` with `tile_shape`
    assert np.all(
            np.pad(a, ((0, 0), (0, 2)), constant_values=9)
            == tile.untile_array(tile.tile_array(a, tile_shape=(2, 4), pad_value=9), array_shape=(6, 8))
    )


def test_untile_masked_array():
    a = np.array([[0, 0, 1, 1],
                  [0, 0, 1, 1],
                  [2, 2, 3, 3],
                  [2, 2, 3, 3]])

    with pytest.raises(AttributeError):
        _ = tile.untile_array(tile.tile_array(a, tile_shape=(2, 2)), array_shape=a.shape).mask

    m = np.array([[False, False, False, True],
                  [False, False, False, False],
                  [False, False, False, False],
                  [False, False, False, True]])

    ma = np.ma.MaskedArray(a, mask=m)
    untiled = tile.untile_array(tile.tile_array(ma.copy(), tile_shape=(2, 2)), array_shape=a.shape)

    assert np.all(ma == untiled)
    assert np.all(ma.mask == untiled.mask)

    untiled = tile.untile_array(tile.tile_array(ma.copy(), tile_shape=(3, 3), pad_value=4), array_shape=a.shape)
    assert np.all(ma == untiled)
    assert np.all(ma.mask == untiled.mask)
