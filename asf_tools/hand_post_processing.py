from pathlib import Path
from tempfile import NamedTemporaryFile

import boto3
import fiona
import numpy as np
import rasterio.windows
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

S3 = boto3.client('s3')

HAND_BUCKET = 'asf-hand-data'
HAND_BUCKET_URI = f'https://{HAND_BUCKET}.s3-us-west-2.amazonaws.com'

DEM_GEOJSON = '/vsicurl/https://asf-dem-west.s3.amazonaws.com/v2/cop30.geojson'
with fiona.open(DEM_GEOJSON) as vds:
    tiles = [feature['properties']['file_path'] for feature in vds]

for tile in tiles:
    tile_id = tile[-22:-8]
    tile_file = Path(tile).name

    water_mask_file = tile_file.replace("COG_", "").replace("_DEM.tif", "_WBM.tif")
    water_mask = f'/vsicurl/{HAND_BUCKET_URI}/WATER_MASKS/{water_mask_file}'

    hand_file = tile_file.replace('_DEM.tif', '_HAND.tif')
    preliminary_hand = f'/vsicurl/{HAND_BUCKET_URI}/GLOBAL_HAND/{tile_id}.tif'

    print(f'PROCESSING: {preliminary_hand}')
    with rasterio.open(tile) as sds:
        dem_bounds = sds.bounds
        dem_meta = sds.meta

    try:
        with rasterio.open(water_mask) as sds:
            window = rasterio.windows.from_bounds(*dem_bounds, sds.transform)
            water_pixels = sds.read(1, window=window)
    except rasterio.RasterioIOError:
        print(f'MISSING: {water_mask}')
        continue

    try:
        with rasterio.open(preliminary_hand) as sds:
            window = rasterio.windows.from_bounds(*dem_bounds, sds.transform)
            out_image = sds.read(1, window=window)
    except rasterio.RasterioIOError:
        print(f'MISSING: {preliminary_hand}')
        continue

    out_image = np.ma.masked_where(water_pixels == 1, out_image)

    with NamedTemporaryFile(suffix='.tif') as tmp_hand:
        with rasterio.open(tmp_hand.name, 'w', **dem_meta) as ods:
            ods.write(np.expand_dims(out_image.filled(0.), axis=0))

        dst_profile = cog_profiles.get("deflate")
        cog_translate(tmp_hand.name, hand_file, dst_profile, in_memory=True)

    print(f'UPLOADING: /vsicurl/{HAND_BUCKET_URI}/GLOBAL_HAND/{hand_file}')
    S3.upload_file(hand_file, HAND_BUCKET, f'GLOBAL_HAND/{hand_file}')

    Path(hand_file).unlink()
