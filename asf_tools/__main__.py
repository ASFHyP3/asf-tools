"""
Creating a water map for RTC products
"""
import argparse
import logging
import sys
from pathlib import Path
from shutil import make_archive

from hyp3lib.aws import upload_file_to_s3
from hyp3lib.image import create_thumbnail
from hyp3lib.makeAsfBrowse import makeAsfBrowse

from asf_tools import water_map


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--bucket', help='AWS bucket to upload product files to')
    parser.add_argument('--bucket-prefix', default='', help='AWS prefix (location in bucket) to add to product files')
    parser.add_argument('vv_raster',
                        help='Sentinel-1 RTC GeoTIFF raster, in power scale, with VV polarization')
    parser.add_argument('vh_raster',
                        help='Sentinel-1 RTC GeoTIFF raster, in power scale, with VH polarization')
    parser.add_argument('--hand-raster',
                        help='Height Above Nearest Drainage (HAND) GeoTIFF aligned to the RTC rasters. '
                             'If not specified, HAND data will be extracted from a Copernicus GLO-30 DEM based HAND.')
    parser.add_argument('--tile-shape', type=int, nargs=2, default=(100, 100),
                        help='shape (height, width) in pixels to tile the image to')
    parser.add_argument('--max-vv-threshold', type=float, default=-15.5,
                        help='Maximum threshold value to use for `vv_raster` in decibels (db)')
    parser.add_argument('--max-vh-threshold', type=float, default=-23.0,
                        help='Maximum threshold value to use for `vh_raster` in decibels (db)')
    parser.add_argument('--hand-threshold', type=float, default=15.,
                        help='The maximum height above nearest drainage in meters to consider a pixel valid')
    parser.add_argument('--hand-fraction', type=float, default=0.8,
                        help='The minimum fraction of valid HAND pixels required in a tile for thresholding')
    parser.add_argument('--membership-threshold', type=float, default=0.45,
                        help='The average membership to the fuzzy indicators required for a water pixel')

    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')
    args = parser.parse_args()

    out_raster = args.vv_raster.replace('_VV.tif', '_WM.tif')
    zip_name = args.vv_raster.replace('_VV.tif', "")

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    logging.debug(' '.join(sys.argv))

    water_map.make_water_map(out_raster, args.vv_raster, args.vh_raster, args.hand_raster, args.tile_shape,
                             args.max_vv_threshold, args.max_vh_threshold, args.hand_threshold, args.hand_fraction,
                             args.membership_threshold)

    logging.info(f'Water map created successfully: {out_raster}')

    parent_directory = Path(out_raster).parent
    output_zip = make_archive(base_name=zip_name, format='zip', base_dir=parent_directory)

    if args.bucket:
        _ = makeAsfBrowse(out_raster, out_raster.replace('.tif', ''), use_nn=True)
        browse_image = out_raster.replace('.tif', '.png')
        thumbnail = create_thumbnail(browse_image)

        upload_file_to_s3(Path(output_zip), args.bucket, args.bucket_prefix)
        upload_file_to_s3(browse_image, args.bucket, args.bucket_prefix)
        upload_file_to_s3(thumbnail, args.bucket, args.bucket_prefix)


if __name__ == '__main__':
    main()
