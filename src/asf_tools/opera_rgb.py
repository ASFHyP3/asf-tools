"""Create a georefernced RGB decomposition RTC image from two input OPERA RTCs"""
import argparse
from pathlib import Path
from typing import Optional

import asf_search
import numpy as np
from osgeo import gdal


gdal.UseExceptions()


def normalize_browse_image_band(
    band_image: np.ndarray, min_percentile: int = 3, max_percentile: int = 97
) -> np.ndarray:
    """Normalize a single band of a browse image to remove outliers.

    Args:
        band_image: A single band numpy array to normalize.
        min_percentile: Lower percentile to clip outliers to.
        max_percentile: Upper percentile to clip outliers to.

    Returns:
        A normalized numpy array.
    """
    vmin = np.nanpercentile(band_image, min_percentile)
    vmax = np.nanpercentile(band_image, max_percentile)

    # gamma correction: 0.5
    is_not_negative = band_image - vmin >= 0
    is_negative = band_image - vmin < 0
    band_image[is_not_negative] = np.sqrt((band_image[is_not_negative] - vmin) / (vmax - vmin))
    band_image[is_negative] = 0
    band_image = np.clip(band_image, 0, 1)
    return band_image


def create_decomposition_rgb(
    copol_path: str, crosspol_path: str, output_path: str, alpha_path: Optional[str] = None
) -> None:
    """Create a georeferenced RGB decomposition RTC image from co-pol and cross-pol RTCs.

    Args:
        copol_path: Path to co-pol RTC.
        crosspol_path: Path to cross-pol RTC.
        output_path: Path to save resulting RGB image to.
        alpha_path: Path to alpha band image. If not provided, the alpha band will be the valid data mask.
    """
    band_list = [None, None, None]

    for filename, is_copol in ((copol_path, True), (crosspol_path, False)):
        gdal_ds = gdal.Open(filename, gdal.GA_ReadOnly)

        gdal_band = gdal_ds.GetRasterBand(1)
        band_image = np.asarray(gdal_band.ReadAsArray(), dtype=np.float32)

        if is_copol:
            # Using copol as template for output
            is_valid = np.isfinite(band_image)
            if alpha_path is None:
                alpha_band = np.asarray(is_valid, dtype=np.float32)
            else:
                alpha_ds = gdal.Open(alpha_path, gdal.GA_ReadOnly)
                alpha_band = np.asarray(alpha_ds.GetRasterBand(1).ReadAsArray(), dtype=np.float32)

            geotransform = gdal_ds.GetGeoTransform()
            projection = gdal_ds.GetProjection()
            shape = band_image.shape
            band_list_index = [0, 2]
        else:
            band_list_index = [1]

        normalized_image = normalize_browse_image_band(band_image)
        for index in band_list_index:
            band_list[index] = normalized_image

    image = np.dstack((band_list[0], band_list[1], band_list[2], alpha_band))

    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_path, shape[1], shape[0], 4, gdal.GDT_Float32)
    output_ds.SetGeoTransform(geotransform)
    output_ds.SetProjection(projection)
    for i in range(4):
        output_ds.GetRasterBand(i + 1).WriteArray(image[:, :, i])


def prep_data(granule: str):
    """Download and prepare the data needed to create an RGB decomposition image.

    Args:
        granule: OPERA granule name.

    Returns:
        Tuple of co-pol, cross-pol, and mask filenames, None for each if not available.
    """
    result = asf_search.granule_search([granule])[0]
    urls = [result.properties['url']]
    others = [x for x in result.properties['additionalUrls'] if 'tif' in x]
    urls += others

    names = [Path(x).name for x in urls]
    copol, crosspol, mask = [None, None, None]

    for name in names:
        image_type = name.split('_')[-1].split('.')[0]
        if image_type in ['VV', 'HH']:
            copol = name

        if image_type in ['VH', 'HV']:
            crosspol = name

        if image_type == 'mask':
            mask = name

    if copol is None or crosspol is None:
        raise ValueError('Both co-pol AND cross-pol data must be available to create an RGB decomposition')

    asf_search.download_urls(urls, path='.')
    return copol, crosspol, mask


def main():
    """Entrypoint for OPERA RTC decomposition image creation."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('granule', nargs=1, default=None, help='OPERA granule name')
    parser.add_argument('--outpath', default='rgb.tif', help='Path to save resulting RGB image to')
    args = parser.parse_args()

    copol, crosspol, mask = prep_data(args.granule[0])
    create_decomposition_rgb(copol, crosspol, args.outpath, mask)
