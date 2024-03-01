import argparse
import os
import time
from pathlib import Path

import numpy as np
from osgeo import gdal

from asf_tools.watermasking.utils import lat_lon_to_tile_string, merge_tiles, remove_temp_files, setup_directories


gdal.UseExceptions()

PREPROCESSED_TILE_DIR = 'worldcover_tiles_preprocessed/'
UNCROPPED_TILE_DIR = 'worldcover_tiles_uncropped/'
CROPPED_TILE_DIR = 'worldcover_tiles/'
FILENAME_POSTFIX = '.tif'
WORLDCOVER_TILE_SIZE = 3
GDAL_OPTIONS = ['COMPRESS=LZW', 'TILED=YES', 'NUM_THREADS=all_cpus']


def tile_preprocessing(tile_dir, min_lat, max_lat, min_lon, max_lon):
    """The worldcover tiles have lots of unnecessary classes, these need to be removed first.
       Note: make a back-up copy of this directory.

    Args:
        tile_dir: The directory containing all of the worldcover tiles.
    """

    filenames = [f for f in os.listdir(tile_dir) if f.endswith('.tif')]

    def filename_filter(filename):
        latitude = int(filename.split('_')[5][1:3])
        longitude = int(filename.split('_')[5][4:7])
        if filename.split('_')[5][3] == 'W':
            longitude = -longitude
        mnlat = min_lat - (min_lat % WORLDCOVER_TILE_SIZE)
        mnlon = min_lon - (min_lon % WORLDCOVER_TILE_SIZE)
        mxlat = max_lat + (max_lat % WORLDCOVER_TILE_SIZE)
        mxlon = max_lon + (max_lon % WORLDCOVER_TILE_SIZE)
        in_lat_range = (latitude >= mnlat) and (latitude <= mxlat)
        in_lon_range = (longitude >= mnlon) and (longitude <= mxlon)
        return in_lat_range and in_lon_range
    filenames_filtered = [f for f in filenames if filename_filter(f)]

    index = 0
    num_tiles = len(filenames_filtered)
    for filename in filenames_filtered:

        start_time = time.time()

        tile_name = filename.split('_')[5]
        filename = str(Path(tile_dir) / filename)
        dst_filename = PREPROCESSED_TILE_DIR + tile_name + '.tif'

        print(f'Processing: {filename}  ---  {dst_filename}  -- {index} of {num_tiles}')

        src_ds = gdal.Open(filename)
        src_band = src_ds.GetRasterBand(1)
        src_arr = src_band.ReadAsArray()

        not_water = np.logical_and(src_arr != 80, src_arr != 0)
        water_arr = np.ones(src_arr.shape)
        water_arr[not_water] = 0

        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            dst_filename,
            water_arr.shape[0],
            water_arr.shape[1],
            1,
            gdal.GDT_Byte,
            options=GDAL_OPTIONS
        )
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())
        dst_band = dst_ds.GetRasterBand(1)
        dst_band.WriteArray(water_arr)
        dst_band.FlushCache()

        del dst_ds
        del src_ds

        end_time = time.time()
        total_time = end_time - start_time

        print(f'Processing {dst_filename} took {total_time} seconds.')

        index += 1


def create_missing_tiles(tile_dir, lat_range, lon_range):
    """Check for, and build, tiles that may be missing from WorldCover, such as over the ocean.

    Args:
        lat_range: The range of latitudes to check.
        lon_range: The range of longitudes to check.
    Returns:
        current_existing_tiles: The list of tiles that exist after the function has completed.
    """
    current_existing_tiles = [f for f in os.listdir(tile_dir) if f.endswith(FILENAME_POSTFIX)]
    for lon in lon_range:
        for lat in lat_range:
            tile = lat_lon_to_tile_string(lat, lon, is_worldcover=True)
            print(f'Checking {tile}')
            if tile not in current_existing_tiles:
                print(f'Could not find {tile}')

                filename = PREPROCESSED_TILE_DIR + tile
                x_size, y_size = 36000, 36000
                x_res, y_res = 8.333333333333333055e-05, -8.333333333333333055e-05
                ul_lon = lon
                ul_lat = lat + WORLDCOVER_TILE_SIZE
                geotransform = (ul_lon, x_res, 0, ul_lat, 0, y_res)

                driver = gdal.GetDriverByName('GTiff')
                ds = driver.Create(
                    filename,
                    xsize=x_size,
                    ysize=y_size,
                    bands=1,
                    eType=gdal.GDT_Byte,
                    options=GDAL_OPTIONS
                )
                ds.SetProjection('EPSG:4326')
                ds.SetGeoTransform(geotransform)
                band = ds.GetRasterBand(1)  # Write ones, as tiles should only be missing over water.
                band.WriteArray(np.ones((x_size, y_size)))

                del ds
                del band

                current_existing_tiles.append(tile)
                print(f'Added {tile}')
    return current_existing_tiles


def get_tiles(osm_tile_coord: tuple, wc_tile_width: int, tile_width: int):
    """Get a list of the worldcover tile locations necessary to fully cover an OSM tile.

    Args:
        osm_tile_coord: The lower left corner coordinate (lat, lon) of the desired OSM tile.
        wc_tile_width: The width/height of the Worldcover tiles in degrees.
        tile_width: The width/height of the OSM tiles in degrees.

    Returns:
        tiles: A list of the lower left corner coordinates of the Worldcover tiles that overlap the OSM tile.
    """

    osm_lat = osm_tile_coord[0]
    osm_lon = osm_tile_coord[1]

    min_lat = osm_lat - (osm_lat % wc_tile_width)
    max_lat = osm_lat + tile_width
    min_lon = osm_lon - (osm_lon % wc_tile_width)
    max_lon = osm_lon + tile_width

    lats = range(min_lat, max_lat, wc_tile_width)
    lons = range(min_lon, max_lon, wc_tile_width)

    tiles = []
    for lat in lats:
        for lon in lons:
            tiles.append((lat, lon))

    return tiles


def lat_lon_to_filenames(worldcover_tile_dir, osm_tile_coord: tuple, wc_tile_width: int, tile_width: int):
    """Get a list of the Worldcover tile filenames that are necessary to overlap an OSM tile.

    Args:
        osm_tile: The lower left corner (lat, lon) of the desired OSM tile.
        wc_tile_width: The width of the Worldcover tiles in degrees.
        tile_width: The width of the OSM tiles in degrees.

    Returns:
        filenames: The list of Worldcover filenames.
    """
    filenames = []
    tiles = get_tiles(osm_tile_coord, wc_tile_width, tile_width)
    for tile in tiles:
        filenames.append(worldcover_tile_dir + lat_lon_to_tile_string(tile[0], tile[1], is_worldcover=True))
    return filenames


def crop_tile(tile, lat, lon, tile_width, tile_height):
    """Crop the merged tiles

    Args:
        tile: The filename of the desired tile to crop.
    """
    in_filename = UNCROPPED_TILE_DIR + tile
    out_filename = CROPPED_TILE_DIR + tile
    pixel_size_x, pixel_size_y = 0.00009009009, -0.00009009009

    src_ds = gdal.Open(in_filename)
    gdal.Translate(
        out_filename,
        src_ds,
        projWin=[lon, lat+tile_height, lon+tile_width, lat],
        xRes=pixel_size_x,
        yRes=pixel_size_y,
        outputSRS='EPSG:4326',
        format='COG',
        creationOptions=['NUM_THREADS=all_cpus']
    )
    remove_temp_files(['tmp_px_size.tif', 'tmp.shp'])


def build_dataset(worldcover_tile_dir, lat_range, lon_range, tile_width, tile_height):
    """ Main function for generating a dataset with worldcover tiles.

    Args:
        worldcover_tile_dir: The directory containing the unprocessed worldcover tiles.
        lat_range: The range of latitudes the dataset should cover.
        lon_range: The range of longitudes the dataset should cover.
        out_degrees: The width of the outputed dataset tiles in degrees.
    """
    for lat in lat_range:
        for lon in lon_range:
            start_time = time.time()
            tile = lat_lon_to_tile_string(lat, lon, is_worldcover=False)
            tile_filename = UNCROPPED_TILE_DIR + tile
            worldcover_tiles = lat_lon_to_filenames(worldcover_tile_dir, (lat, lon), WORLDCOVER_TILE_SIZE, tile_width)
            print(f'Processing: {tile_filename} {worldcover_tiles}')
            merge_tiles(worldcover_tiles, tile_filename, 'GTiff', compress=True)
            crop_tile(tile, lat, lon, tile_width, tile_height)
            end_time = time.time()
            total_time = end_time - start_time
            print(f'Time Elapsed: {total_time}s')


def main():

    parser = argparse.ArgumentParser(
        prog='generate_worldcover_tiles.py',
        description='Main script for creating a tiled watermask dataset from the ESA WorldCover dataset.'
    )

    parser.add_argument('--worldcover-tiles-dir', help='The path to the directory containing the worldcover tifs.')
    parser.add_argument(
        '--lat-begin',
        help='The minimum latitude of the dataset in EPSG:4326.',
        default=-85,
        required=True
    )
    parser.add_argument('--lat-end', help='The maximum latitude of the dataset in EPSG:4326.', default=85)
    parser.add_argument('--lon-begin', help='The minimum longitude of the dataset in EPSG:4326.', default=-180)
    parser.add_argument('--lon-end', help='The maximum longitude of the dataset in EPSG:4326.', default=180)
    parser.add_argument('--tile-width', help='The desired width of the tile in degrees.', default=5)
    parser.add_argument('--tile-height', help='The desired height of the tile in degrees.', default=5)

    args = parser.parse_args()

    lat_begin = int(args.lat_begin)
    lat_end = int(args.lat_end)
    lon_begin = int(args.lon_begin)
    lon_end = int(args.lon_end)
    tile_width = int(args.tile_width)
    tile_height = int(args.tile_height)
    lat_range = range(lat_begin, lat_end, tile_width)
    lon_range = range(lon_begin, lon_end, tile_height)

    setup_directories([PREPROCESSED_TILE_DIR, UNCROPPED_TILE_DIR, CROPPED_TILE_DIR])

    # Process the multi-class masks into water/not-water masks.
    tile_preprocessing(args.worldcover_tiles_dir, lat_begin, lat_end, lon_begin, lon_end)

    wc_lat_range = range(
        lat_begin - (lat_begin % WORLDCOVER_TILE_SIZE),
        lat_end + (lat_end % WORLDCOVER_TILE_SIZE),
        WORLDCOVER_TILE_SIZE
    )
    wc_lon_range = range(
        lon_begin - (lon_begin % WORLDCOVER_TILE_SIZE),
        lon_end + (lon_end % WORLDCOVER_TILE_SIZE),
        WORLDCOVER_TILE_SIZE
    )

    # Ocean only tiles are missing from WorldCover, so we need to create blank (water-only) ones.
    create_missing_tiles(PREPROCESSED_TILE_DIR, wc_lat_range, wc_lon_range)

    build_dataset(
        PREPROCESSED_TILE_DIR,
        lat_range,
        lon_range,
        tile_width=tile_width,
        tile_height=tile_height
    )


if __name__ == '__main__':
    main()
