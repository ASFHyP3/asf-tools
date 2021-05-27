import logging
import warnings
from pathlib import Path
from typing import Literal, Union

import numpy as np
from osgeo import gdal

gdal.UseExceptions()
log = logging.getLogger(__name__)


def convert_scale(array: Union[np.ndarray, np.ma.MaskedArray], in_scale: Literal['db', 'amplitude', 'power'],
                  out_scale: Literal['db', 'amplitude', 'power']) -> Union[np.ndarray, np.ma.MaskedArray]:
    """Convert calibrated raster scale between db, amplitude and power"""
    if in_scale == out_scale:
        warnings.warn(f'Nothing to do! {in_scale} is same as {out_scale}.')
        return array

    log10 = np.ma.log10 if isinstance(array, np.ma.MaskedArray) else np.log10

    if in_scale == 'db':
        if out_scale == 'power':
            return 10 ** (array / 10)
        if out_scale == 'amplitude':
            return 10 ** (array / 20)

    if in_scale == 'amplitude':
        if out_scale == 'power':
            return array ** 2
        if out_scale == 'db':
            return 10 * log10(array ** 2)

    if in_scale == 'power':
        if out_scale == 'amplitude':
            return np.sqrt(array)
        if out_scale == 'db':
            return 10 * log10(array)

    raise ValueError(f'Cannot convert raster of scale {in_scale} to {out_scale}')


def read_as_masked_array(raster: Union[str, Path], band: int = 1) -> np.ma.MaskedArray:
    """Reads data from a raster image into memory, masking invalid and NoData values

    Args:
        raster: The file path to a raster image
        band: The raster band to read

    Returns:
        data: The raster pixel data as a numpy MaskedArray
    """
    log.debug(f'Reading raster values from {raster}')
    ds = gdal.Open(str(raster))
    band = ds.GetRasterBand(band)
    data = np.ma.masked_invalid(band.ReadAsArray())
    nodata = band.GetNoDataValue()
    if nodata is not None:
        return np.ma.masked_values(data, nodata)
    return data
