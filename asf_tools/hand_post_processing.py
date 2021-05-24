"""Post-process HAND data to: Subset to DEM COG, set HAND values to zero for ocean pixels, and COG-ify"""
import argparse
import logging
import multiprocessing
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import boto3
import fiona
import numpy as np
import rasterio.enums
import rasterio.windows
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

S3 = boto3.client('s3')

HAND_BUCKET = 'asf-hand-data'
HAND_BUCKET_URI = f'https://{HAND_BUCKET}.s3-us-west-2.amazonaws.com'
DEM_GEOJSON = '/vsicurl/https://asf-dem-west.s3.amazonaws.com/v2/cop30.geojson'


def post_process_hand_for_dem_tile(dem_tile: str):
    logger = multiprocessing.get_logger()

    tile_id = dem_tile[-22:-8]
    tile_file = Path(dem_tile).name

    water_mask_file = tile_file.replace("COG_", "").replace("_DEM.tif", "_WBM.tif")
    water_mask = f'/vsicurl/{HAND_BUCKET_URI}/WATER_MASKS/{water_mask_file}'

    hand_file = tile_file.replace('_DEM.tif', '_HAND.tif')
    preliminary_hand = f'/vsicurl/{HAND_BUCKET_URI}/GLOBAL_HAND/{tile_id}.tif'
    hand = f'/vsicurl/{HAND_BUCKET_URI}/GLOBAL_HAND/{hand_file}'

    logger.info(f'PROCESSING: {preliminary_hand}')

    try:
        with rasterio.open(dem_tile) as sds:
            dem_bounds = sds.bounds
            dem_meta = sds.meta

        try:
            with rasterio.open(water_mask) as sds:
                window = rasterio.windows.Window(0, 0, height=dem_meta['height'], width=dem_meta['width'])
                water_pixels = sds.read(1, window=window)
        except rasterio.RasterioIOError:
            logger.info(f'MISSING: {water_mask}')
            return

        try:
            with rasterio.open(preliminary_hand) as sds:
                window = rasterio.windows.from_bounds(*dem_bounds, sds.transform)
                out_image = sds.read(
                    1, window=window, out_shape=(dem_meta['height'], dem_meta['width']),
                    resampling=rasterio.enums.Resampling.bilinear
                )
        except rasterio.RasterioIOError:
            logger.info(f'MISSING: {preliminary_hand}')
            return

        out_image = np.ma.masked_where(water_pixels == 1, out_image)

        with NamedTemporaryFile(suffix='.tif') as tmp_hand:
            with rasterio.open(tmp_hand.name, 'w', **dem_meta) as ods:
                ods.write(np.expand_dims(out_image.filled(0.), axis=0))

            dst_profile = cog_profiles.get("deflate")
            cog_translate(tmp_hand.name, hand_file, dst_profile, in_memory=True)

        logger.info(f'UPLOADING: {hand}')
        S3.upload_file(hand_file, HAND_BUCKET, f'GLOBAL_HAND/{hand_file}')

        Path(hand_file).unlink()
    except Exception as e:  # noqa: B902
        logger.info(f'WOOPS: {preliminary_hand} {water_mask} {dem_tile} :{e}')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-p', '--processes', type=int, default=None, help='Number of processors to use')
    parser.add_argument('-v', '--verbose', action='store_true', help='Turn on verbose logging')
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(level)
    logger.info(' '.join(sys.argv))

    with fiona.open(DEM_GEOJSON) as vds:
        tiles = [feature['properties']['file_path'] for feature in vds]

    p = multiprocessing.Pool(processes=args.processes)
    p.map(post_process_hand_for_dem_tile, tiles)
    p.close()
    p.join()


if __name__ == '__main__':
    main()
