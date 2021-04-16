import numpy as np

import pytest

from asf_tools import raster


def test_convert_scale():
    c = raster.convert_scale(np.array([-10, -5, 0, 5, 10]), 'amplitude', 'power')
    assert np.all(np.isclose(np.array([100, 25, 0, 25, 100]), c))

    c = raster.convert_scale(np.array([-10, -5, 0, 5, 10]), 'amplitude', 'db')
    assert np.all(np.isclose(np.array([20., 13.97940009, -np.inf, 13.97940009, 20.]), c))

    c = raster.convert_scale(np.array([-1, 0, 1, 4, 9]), 'power', 'amplitude')
    assert np.isnan(c[0])
    assert np.all(np.isclose(np.array([0, 1, 2, 3]), c[1:]))

    c = raster.convert_scale(np.array([-1, 0, 1, 4, 9]), 'power', 'db')
    assert np.isnan(c[0])
    assert np.all(np.isclose(np.array([-np.inf, 0., 6.02059991, 9.54242509]), c[1:]))

    c = raster.convert_scale(np.array([np.nan, -np.inf, 0., 6.02059991, 9.54242509]), 'db', 'power')
    assert np.isnan(c[0])
    assert np.all(np.isclose(np.array([0, 1, 4, 9]), c[1:]))

    c = raster.convert_scale(np.array([-np.inf, -20., 0., 13.97940009, 20.]), 'db', 'amplitude')
    assert np.all(np.isclose(np.array([0., 0.1, 1., 5., 10.]), c))

    a = np.array([-10, -5, 0, 5, 10])
    with pytest.raises(ValueError):
        _ = raster.convert_scale(a, 'power', 'foo')
    with pytest.raises(ValueError):
        _ = raster.convert_scale(a, 'bar', 'amplitude')

    with pytest.warns(UserWarning):
        assert np.all(np.array([-10, -5, 0, 5, 10]) == raster.convert_scale(a, 'amplitude', 'amplitude'))
    with pytest.warns(UserWarning):
        assert np.all(np.array([-10, -5, 0, 5, 10]) == raster.convert_scale(a, 'power', 'power'))
    with pytest.warns(UserWarning):
        assert np.all(np.array([-10, -5, 0, 5, 10]) == raster.convert_scale(a, 'db', 'db'))
