import argparse
import os
import subprocess

import numpy as np
from osgeo import gdal, osr

from asf_tools.watermasking.utils import lat_lon_to_tile_string


gdal.UseExceptions()


def main():

    parser = argparse.ArgumentParser(
        prog='fill_missing_tiles.py',
        description='Script for creating filled tifs in areas with missing tiles.'
    )

    parser.add_argument('--fill-value', help='The value to fill the data array with.', default=0)
    parser.add_argument('--lat-begin', help='The minimum latitude of the dataset in EPSG:4326.', default=-85)
    parser.add_argument('--lat-end', help='The maximum latitude of the dataset in EPSG:4326.', default=85)
    parser.add_argument('--lon-begin', help='The minimum longitude of the dataset in EPSG:4326.', default=-180)
    parser.add_argument('--lon-end', help='The maximum longitude of the dataset in EPSG:4326.', default=180)
    parser.add_argument('--tile-width', help='The desired width of the tile in degrees.', default=5)
    parser.add_argument('--tile-height', help='The desired height of the tile in degrees.', default=5)

    args = parser.parse_args()

    fill_value = int(args.fill_value)
    lat_begin = int(args.lat_begin)
    lat_end = int(args.lat_end)
    lon_begin = int(args.lon_begin)
    lon_end = int(args.lon_end)
    tile_width = int(args.tile_width)
    tile_height = int(args.tile_height)

    lat_range = range(lat_begin, lat_end, tile_height)
    lon_range = range(lon_begin, lon_end, tile_width)

    for lat in lat_range:
        for lon in lon_range:

            tile = lat_lon_to_tile_string(lat, lon, is_worldcover=False, postfix='')
            tile_tif = 'tiles/' + tile + '.tif'
            tile_cog = 'tiles/cogs/' + tile + '.tif'

            print(f'Processing: {tile}')

            xmin, ymin = lon, lat
            pixel_size_x = 0.00009009009
            pixel_size_y = 0.00009009009

            # All images in the dataset should be this size.
            data = np.empty((55500, 55500))
            data.fill(fill_value)

            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(tile_tif, xsize=data.shape[0], ysize=data.shape[1], bands=1, eType=gdal.GDT_Byte)
            dst_ds.SetGeoTransform([xmin, pixel_size_x, 0, ymin, 0, pixel_size_y])
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            dst_ds.SetProjection(srs.ExportToWkt())
            dst_band = dst_ds.GetRasterBand(1)
            dst_band.WriteArray(data)
            del dst_ds

            command = f'gdal_translate -of COG -co NUM_THREADS=all_cpus {tile_tif} {tile_cog}'.split(' ')
            subprocess.run(command)
            os.remove(tile_tif)


if __name__ == '__main__':
    main()
