import numpy as np
import pytest

from asf_tools import raster


def test_convert_scale():
    c = raster.convert_scale(np.array([-10, -5, 0, 5, 10]), 'amplitude', 'power')
    assert np.allclose(c, np.array([100, 25, 0, 25, 100]))

    c = raster.convert_scale(np.array([-10, -5, 0, 5, 10]), 'amplitude', 'db')
    assert np.allclose(c, np.array([20., 13.97940009, -np.inf, 13.97940009, 20.]))

    c = raster.convert_scale(np.array([-1, 0, 1, 4, 9]), 'power', 'amplitude')
    assert np.isnan(c[0])
    assert np.allclose(c[1:], np.array([0, 1, 2, 3]))

    c = raster.convert_scale(np.array([-1, 0, 1, 4, 9]), 'power', 'db')
    assert np.isnan(c[0])
    assert np.allclose(c[1:], np.array([-np.inf, 0., 6.02059991, 9.54242509]), )

    c = raster.convert_scale(np.array([np.nan, -np.inf, 0., 6.02059991, 9.54242509]), 'db', 'power')
    assert np.isnan(c[0])
    assert np.allclose(c[1:], np.array([0, 1, 4, 9]))

    c = raster.convert_scale(np.array([-np.inf, -20., 0., 13.97940009, 20.]), 'db', 'amplitude')
    assert np.allclose(c, np.array([0., 0.1, 1., 5., 10.]))

    a = np.array([-10, -5, 0, 5, 10])
    with pytest.raises(ValueError):
        _ = raster.convert_scale(a, 'power', 'foo')
    with pytest.raises(ValueError):
        _ = raster.convert_scale(a, 'bar', 'amplitude')

    with pytest.warns(UserWarning):
        assert np.allclose(raster.convert_scale(a, 'amplitude', 'amplitude'), np.array([-10, -5, 0, 5, 10]))
    with pytest.warns(UserWarning):
        assert np.allclose(raster.convert_scale(a, 'power', 'power'), np.array([-10, -5, 0, 5, 10]))
    with pytest.warns(UserWarning):
        assert np.allclose(raster.convert_scale(a, 'db', 'db'), np.array([-10, -5, 0, 5, 10]))


def test_convert_scale_masked_arrays():
    masked_array = np.ma.MaskedArray([-1, 0, 1, 4, 9], mask=[False, False, False, False, False])
    c = raster.convert_scale(masked_array, 'power', 'db')
    assert np.allclose(c.mask, [True, True, False, False, False])
    assert np.allclose(
        c, np.ma.MaskedArray([np.nan, -np.inf, 0., 6.02059991, 9.54242509], mask=[True, True, False, False, False])
    )

    a = raster.convert_scale(c, 'db', 'power')
    assert np.allclose(a.mask, [True, True, False, False, False])
    assert np.allclose(a, np.ma.MaskedArray([-1, 0, 1, 4, 9], mask=[True, True, False, False, False]))
