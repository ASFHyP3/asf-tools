import argparse
import logging
import sys
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import boto3
from botocore.exceptions import ClientError
import numpy as np
import rasterio
from shapely.geometry import GeometryCollection, box

from asf_tools.composite import write_cog
from asf_tools.dem import prepare_dem_vrt
from asf_tools.hand.calculate import calculate_hand_for_basins
# from asf_tools.hand.upload_to_s3 import upload_file

log = logging.getLogger(__name__)

#S3 = boto3.client('s3')

def upload_file(file_name, bucket, obj_name):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param obj_name: the S3 object name. If not specified then file_name is used

    :return: True if file was uploaded, else False
    """
    base_name = os.path.basename(file_name)
    # If S3 object_name was not specified, use file_name
    if obj_name is None:
        obj_name = base_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        with open(file_name, "rb") as f:
            s3_client.upload_fileobj(f, bucket, obj_name)

    except ClientError as e:
        logging.error(e)
        return False
    return True


def dem_to_aux_path(dem_path: str, aux_type: Literal['EDM', 'FLM', 'HEM', 'WBM'] = 'WBM') -> str:
    aux_path = dem_path.replace('DEM/Copernicus', 'DEM/AUXFILES/Copernicus')
    aux_path = aux_path.replace('_DEM.tif', f'_{aux_type}.tif')
    return aux_path


def make_asf_hand(dem_tile: str, buffer: float = 0.5) -> str:
    log.info(f'Buffering DEM by {buffer} degrees to avoid clipping basins')
    with rasterio.open(dem_tile) as dem:
        dem_bounds = dem.bounds
        dem_meta = dem.meta

    dem_geometry = box(*dem_bounds)
    dem_buffered = GeometryCollection([dem_geometry.buffer(buffer)])

    with NamedTemporaryFile(suffix='.vrt', delete=False) as buffered_dem_vrt:
        prepare_dem_vrt(buffered_dem_vrt.name, dem_buffered)

    log.info('Calculating initial HAND for buffered DEM')
    with NamedTemporaryFile(suffix='.tif', delete=False) as hand_raster:
        calculate_hand_for_basins(hand_raster.name, dem_buffered, buffered_dem_vrt.name)

    log.info('Cropping buffered HAND to original DEM size')
    with rasterio.open(hand_raster.name) as sds:
        window = rasterio.windows.from_bounds(*dem_bounds, sds.transform)
        out_pixels = sds.read(
            1, window=window, out_shape=(dem_meta['height'], dem_meta['width']),
            resampling=rasterio.enums.Resampling.bilinear
        )

    log.info('Masking ocean pixels as identified in WBM auxiliary DEM file')
    wmb_tile = dem_to_aux_path(dem_tile)
    with rasterio.open(wmb_tile) as wbm:
        wbm_pixels = wbm.read(1)

    out_pixels = np.ma.masked_where(wbm_pixels == 1, out_pixels)

    log.info('Finalizing HAND tile')
    out_raster = Path(dem_tile).name.replace('DEM.tif', 'HAND.tif')
    write_cog(out_raster, out_pixels, transform=dem.transform.to_gdal(), epsg_code=dem.crs.to_epsg())

    return out_raster


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('dem_tile', help='VSICURL path to a Copernicus GLO-30 DEM GeoTIFF to calculate HAND from. '
                                         'Tile must be located in the AWS Open Data bucket `s3://copernicus-dem-30m/`. '
                                         'For more info, see: https://registry.opendata.aws/copernicus-dem/')

    parser.add_argument('-b', '--s3-bucket', help='Upload tile to this AWS S3 Bucket when done')
    parser.add_argument('-p', '--s3-prefix', help='Upload tile under this AWS S3 prefix when done. '
                                                  'Requires `--s3-bucket` option to also be provided.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')
    args = parser.parse_args()

    if args.s3_prefix is not None and args.s3_bucket is None:
        parser.error('`--s3-prefix` may only be used in conjunction with `--s3-bucket`')

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    log.debug(' '.join(sys.argv))
    log.info(f'Making HAND tile from {args.dem_tile}')

    hand_raster = make_asf_hand(args.dem_tile)

    log.info(f'HAND tile COG created successfully: {hand_raster}')
    hand_raster_tile_name = hand_raster.replace(".tif", "")

    if args.s3_bucket:
        key = f'{args.s3_prefix}/{hand_raster_tile_name}/{hand_raster}' if args.s3_prefix else str(hand_raster)
        log.info(f'Uploading HAND tile to s3://{args.s3_bucket}/{key}')
        upload_file(hand_raster, args.s3_bucket, key)
        #S3.upload_file(hand_raster, args.s3_bucket, key)
        # NOTE: I'd like the new HAND we calculate to be laid out exactly like the COP DEM bucket. So, if you
        #       know the COP DEM S3 path/url all you'd have to do is replace the bucket name and `_DEM` with `_HAND`


if __name__ == '__main__':
    main()
