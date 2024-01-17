"""Create a georefernced RGB decomposition RTC image from two input OPERA RTCs"""
import argparse
from pathlib import Path

import asf_search
import numpy as np
from osgeo import gdal


gdal.UseExceptions()

BROWSE_IMAGE_MIN_PERCENTILE = 3
BROWSE_IMAGE_MAX_PERCENTILE = 97


def normalize_browse_image_band(band_image):
    vmin = np.nanpercentile(band_image, BROWSE_IMAGE_MIN_PERCENTILE)
    vmax = np.nanpercentile(band_image, BROWSE_IMAGE_MAX_PERCENTILE)

    # gamma correction: 0.5
    is_not_negative = band_image - vmin >= 0
    is_negative = band_image - vmin < 0
    band_image[is_not_negative] = np.sqrt((band_image[is_not_negative] - vmin) / (vmax - vmin))
    band_image[is_negative] = 0
    band_image = np.clip(band_image, 0, 1)
    return band_image


def create_browse_imagery(copol_path, crosspol_path, output_path, alpha_channel=None):
    band_list = [None, None, None]

    for filename, is_copol in ((copol_path, True), (crosspol_path, False)):
        gdal_ds = gdal.Open(filename, gdal.GA_ReadOnly)

        gdal_band = gdal_ds.GetRasterBand(1)
        band_image = np.asarray(gdal_band.ReadAsArray(), dtype=np.float32)

        if is_copol:
            # Using copol as template for output
            is_valid = np.isfinite(band_image)
            if alpha_channel is None:
                alpha_channel = np.asarray(is_valid, dtype=np.float32)
            geotransform = gdal_ds.GetGeoTransform()
            projection = gdal_ds.GetProjection()
            shape = band_image.shape
            band_list_index = [0, 2]
        else:
            band_list_index = [1]

        normalized_image = normalize_browse_image_band(band_image)
        for index in band_list_index:
            band_list[index] = normalized_image

    image = np.dstack((band_list[0], band_list[1], band_list[2], alpha_channel))

    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_path, shape[1], shape[0], 4, gdal.GDT_Float32)
    output_ds.SetGeoTransform(geotransform)
    output_ds.SetProjection(projection)
    for i in range(4):
        output_ds.GetRasterBand(i + 1).WriteArray(image[:, :, i])


def prep_data(granule):
    result = asf_search.granule_search([granule])[0]
    urls = [result.properties['url']]
    others = [x for x in result.properties['additionalUrls'] if 'tif' in x]
    urls += others
    asf_search.download_urls(urls, path='.')
    names = [Path(x).name for x in urls]
    for name in names:
        image_type = name.split('_')[-1].split('.')[0]
        if image_type in ['VV', 'HH']:
            copol = name
        elif image_type in ['VH', 'HV']:
            crosspol = name
        elif image_type == 'mask':
            mask = name
    return copol, crosspol, mask


def main():
    """Entrypoint for OPERA RTC decomposition image creation."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('granule', nargs=1, default=None, help='OPERA granule name')
    parser.add_argument('--outpath', default='rgb.tif', help='Path to save resulting RGB image to')
    args = parser.parse_args()

    copol, crosspol, _ = prep_data(args.granule[0])
    create_browse_imagery(Path(copol), Path(crosspol), Path(args.outpath))
