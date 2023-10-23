import logging
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal, Union

import numpy as np
from osgeo import gdal

from asf_tools.util import epsg_to_wkt

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
    del ds  # How to close w/ gdal
    return data


def read_as_array(raster: str, band: int = 1) -> np.array:
    """Reads data from a raster image into memory

    Args:
        raster: The file path to a raster image
        band: The raster band to read

    Returns:
        data: The raster pixel data as a numpy array
    """
    log.debug(f'Reading raster values from {raster}')
    ds = gdal.Open(raster)
    data = ds.GetRasterBand(band).ReadAsArray()
    del ds  # How to close w/ gdal
    return data


def write_cog(file_name: Union[str, Path], data: np.ndarray, transform: List[float], epsg_code: int,
              dtype=gdal.GDT_Float32, nodata_value=None):
    """Creates a Cloud Optimized GeoTIFF

    Args:
        file_name: The output file name
        data: The raster data
        transform: The geotransform for the output GeoTIFF
        epsg_code: The integer EPSG code for the output GeoTIFF projection
        dtype: The pixel data type for the output GeoTIFF
        nodata_value: The NODATA value for the output Geotiff

    Returns:
        file_name: The output file name
    """
    log.info(f'Creating {file_name}')

    with NamedTemporaryFile() as temp_file:
        driver = gdal.GetDriverByName('GTiff')
        temp_geotiff = driver.Create(temp_file.name, data.shape[1], data.shape[0], 1, dtype)
        temp_geotiff.GetRasterBand(1).WriteArray(data)
        if nodata_value is not None:
            temp_geotiff.GetRasterBand(1).SetNoDataValue(nodata_value)
        temp_geotiff.SetGeoTransform(transform)
        temp_geotiff.SetProjection(epsg_to_wkt(epsg_code))

        driver = gdal.GetDriverByName('COG')
        options = ['COMPRESS=LZW', 'OVERVIEW_RESAMPLING=AVERAGE', 'NUM_THREADS=ALL_CPUS', 'BIGTIFF=YES']
        driver.CreateCopy(str(file_name), temp_geotiff, options=options)

        del temp_geotiff  # How to close w/ gdal
    return file_name
