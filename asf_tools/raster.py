import warnings
from typing import Literal

import numpy as np


def convert_scale(array: np.ndarray, in_scale: Literal['db', 'amplitude', 'power'],
                  out_scale: Literal['db', 'amplitude', 'power']) -> np.ndarray:
    """Convert calibrated raster scale between db, amplitude and power"""
    if in_scale == out_scale:
        warnings.warn(f'Nothing to do! {in_scale} is same as {out_scale}.')
        return array

    if in_scale == 'db':
        if out_scale == 'power':
            return 10 ** (array / 10)
        if out_scale == 'amplitude':
            return 10 ** (array / 20)

    if in_scale == 'amplitude':
        if out_scale == 'power':
            return array ** 2
        if out_scale == 'db':
            return 10 * np.log10(array ** 2)

    if in_scale == 'power':
        if out_scale == 'amplitude':
            return np.sqrt(array)
        if out_scale == 'db':
            return 10 * np.log10(array)

    raise ValueError(f'Cannot convert raster of scale {in_scale} to {out_scale}')
