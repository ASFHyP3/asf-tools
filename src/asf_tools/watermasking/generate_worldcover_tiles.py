import argparse
import os
import time

import numpy as np

from osgeo import gdal

from asf_tools.watermasking.utils import lat_lon_to_tile_string, remove_temp_files, merge_tiles

TILE_DIR  = 'worldcover_tiles_unfinished/'
CROPPED_TILE_DIR = 'worldcover_tiles/'
FILENAME_POSTFIX = '.tif'
WORLDCOVER_TILE_SIZE = 3


def tile_preprocessing(tile_dir, min_lat, max_lat):
    """The worldcover tiles have lots of unnecessary classes, these need to be removed first.
       Note: make a back-up copy of this directory.

    Args:
        tile_dir: The directory containing all of the worldcover tiles.
    """

    filenames = [f for f in os.listdir(tile_dir) if f.endswith('.tif')]
    filter = lambda filename: (int(filename.split('_')[5][1:3]) >= min_lat) and (int(filename.split('_')[5][1:3]) <= max_lat)
    filenames_filtered = [f for f in filenames if filter(f)] 

    index = 0
    num_tiles = len(filenames_filtered)
    for filename in filenames_filtered:
        start_time = time.time()

        dst_filename = filename.split('_')[5] + '.tif'

        print(f'Processing: {filename}  ---  {dst_filename}  -- {index} of {num_tiles}')

        src_ds = gdal.Open(filename)
        src_band = src_ds.GetRasterBand(1)
        src_arr = src_band.ReadAsArray()

        not_water = np.logical_and(src_arr != 80, src_arr != 0)
        water_arr = np.ones(src_arr.shape)
        water_arr[not_water] = 0

        driver = gdal.GetDriverByName('GTiff')

        dst_ds = driver.Create(dst_filename, water_arr.shape[0], water_arr.shape[1], 1, gdal.GDT_Byte, options=['COMPRESS=LZW', 'TILED=YES'])
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())
        dst_band = dst_ds.GetRasterBand(1)
        dst_band.WriteArray(water_arr)
        dst_band.FlushCache()

        del dst_ds
        del src_ds

        end_time = time.time()
        total_time = end_time - start_time # seconds

        print(f'Processing {dst_filename} took {total_time} seconds.')

        index += 1


def get_tiles(osm_tile_coord: tuple, wc_deg: int, osm_deg: int):
    """Get a list of the worldcover tile locations necessary to fully cover an OSM tile.
    
    Args:
        osm_tile_coord: The lower left corner coordinate (lat, lon) of the desired OSM tile.
        wc_deg: The width/height of the Worldcover tiles in degrees.
        osm_deg: The width/height of the OSM tiles in degrees.
    
    Returns:
        tiles: A list of the lower left corner coordinates of the Worldcover tiles that overlap the OSM tile.
    """

    osm_lat = osm_tile_coord[0]
    osm_lon = osm_tile_coord[1]

    min_lat = osm_lat - (osm_lat % wc_deg)
    max_lat = osm_lat + osm_deg
    min_lon = osm_lon - (osm_lon % wc_deg)
    max_lon = osm_lon + osm_deg

    lats = range(min_lat, max_lat, wc_deg)
    lons = range(min_lon, max_lon, wc_deg)

    tiles = []
    for lat in lats:
        for lon in lons:
            tiles.append((lat, lon))

    return tiles


def lat_lon_to_filenames(osm_tile_coord: tuple, wc_deg: int, osm_deg: int):
    """Get a list of the Worldcover tile filenames that are necessary to overlap an OSM tile.

    Args:
        osm_tile: The lower left corner (lat, lon) of the desired OSM tile.
        wc_deg: The width of the Worldcover tiles in degrees.
        osm_deg: The width of the OSM tiles in degrees.
    
    Returns:
        filenames: The list of Worldcover filenames.
    """
    filenames = []
    tiles = get_tiles(osm_tile_coord, wc_deg, osm_deg)
    for tile in tiles:
        filenames.append(lat_lon_to_tile_string(tile[0], tile[1], is_worldcover=True))
    return filenames


def crop_tile(tile):
    """Crop the merged tiles"""
    try:
        ref_image = TILE_DIR + tile
        pixel_size = gdal.Warp('tmp_px_size.tif', ref_image, dstSRS='EPSG:4326').GetGeoTransform()[1]
        shapefile_command = ' '.join(['gdaltindex', 'tmp.shp', ref_image])
        os.system(shapefile_command)
        out_filename = CROPPED_TILE_DIR + tile
        gdal.Warp(
            out_filename,
            tile,
            cutlineDSName='tmp.shp',
            cropToCutline=True,
            xRes=pixel_size,
            yRes=pixel_size,
            targetAlignedPixels=True,
            dstSRS='EPSG:4326',
            format='COG'
        )
        remove_temp_files(['tmp_px_size.tif', 'tmp.shp'])
    except Exception as e:  # When a tile fails to process, we don't necessarily want the program to stop.
        print(f'Could not process {tile}. Continuing...')
    index += 1


def build_dataset(lat_range, lon_range, worldcover_degrees, osm_degrees):
    for lat in lat_range:
        for lon in lon_range:
            start_time = time.time()
            tile_filename = TILE_DIR + lat_lon_to_tile_string(lat, lon, is_worldcover=False)
            worldcover_tiles = lat_lon_to_filenames(lat, lon, worldcover_degrees, osm_degrees)
            print(f'Processing: {tile_filename} {worldcover_tiles}') 
            merge_tiles(worldcover_tiles, tile_filename)
            end_time = time.time()
            total_time = end_time - start_time
            print(f'Time Elapsed: {total_time}s')


def main():
    parser = argparse.ArgumentParser(
        prog='generate_worldcover_tiles.py',
        description='Main script for creating a tiled watermask dataset from the ESA WorldCover dataset.'
    )
    parser.add_argument('--worldcover-tiles-dir', help='The path to the directory containing the worldcover tifs.')
    parser.add_argument('--lat-begin', help='The minimum latitude of the dataset.', default=-85, required=True)
    parser.add_argument('--lat-end', help='The maximum latitude of the dataset.', default=85)
    parser.add_argument('--lon-begin', help='The minimum longitude of the dataset.', default=-180)
    parser.add_argument('--lon-end', help='The maximum longitude of the dataset.', default=180)
    parser.add_argument('--tile-width', help='The desired width of the tile in degrees.', default=5)
    parser.add_argument('--tile-height', help='The desired height of the tile in degrees.', default=5)

    args = parser.parse_args()

    for dir in [TILE_DIR, CROPPED_TILE_DIR]:
        try:
            os.mkdir(dir)
        except FileExistsError as e:
            print(f'{dir} already exists. Skipping...')

    tile_preprocessing(args.worldcover_tiles_dir, args.lat_begin, args.lat_end)

    lat_range = range(args.lat_begin, args.lat_end, args.tile_width)
    lon_range = range(args.lon_begin, args.lon_end, args.tile_heigth)

    build_dataset(lat_range, lon_range, worldcover_degrees=WORLDCOVER_TILE_SIZE, osm_degrees=args.tile_width)


if __name__ == '__main__':
    main()