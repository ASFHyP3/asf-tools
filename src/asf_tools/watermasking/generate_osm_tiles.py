import argparse
import os
import time

import geopandas as gpd
import numpy as np
from osgeo import gdal

from asf_tools.watermasking.utils import lat_lon_to_tile_string

INTERIOR_TILE_DIR = 'interior_tiles/'
OCEAN_TILE_DIR = 'ocean_tiles/'
FINISHED_TILE_DIR = 'tiles/'


def process_pbf(planet_file: str, output_file: str):
    """Process the global PBF file so that it only contains water features.

    Args:
        planet_file: The path to the OSM Planet PBF file.
        output_file: The desired path of the processed PBF file. 
    """

    natural_file = 'planet_natural.pbf'
    waterways_file = 'planet_waterways.pbf'
    reservoirs_file = 'planet_reservoirs.pbf'

    natural_water_command = f'osmium tags-filter -o {natural_file} {planet_file} wr/natural=water'
    waterways_command = f'osmium tags-filter -o {waterways_file} {planet_file} waterway="*"'
    reservoirs_command = f'osmium tags-filter -o {reservoirs_file} {planet_file} landuse=reservoir'
    merge_command = f'osmium merge {natural_file} {waterways_file} {reservoirs_file} -o {output_file}'

    os.system(natural_water_command)
    os.system(waterways_command)
    os.system(reservoirs_command)
    os.system(merge_command)


def remove_temp_files(temp_files: list):
    """Remove each file in a list of files.
    
    Args:
        temp_files: The list of temporary files to remove.
    """
    for file in temp_files:
        try:
            os.remove(file)
        except:
            print('Error removing temporary file. Skipping it...')


def process_ocean_tiles(ocean_polygons_path, lat, lon, tile_width_deg, tile_height_deg, output_dir):
    """Process and crop OSM ocean polygons into a tif tile.
    
    Args:
        ocean_polygons_path: The path to the global ocean polygons file from OSM.
        lat: The minimum latitude of the tile.
        lon: The minimum longitude of the tile.
        tile_width_deg: The width of the tile in degrees.
        tile_height_deg: The height of the tile in degrees.
    """

    tile = lat_lon_to_tile_string(lat, lon, is_worldcover=False, postfix='')
    tile_tif = output_dir + tile + '.tif'

    xmin, xmax, ymin, ymax = lon, lon+tile_width_deg, lat, lat+tile_height_deg
    pixel_size_x = 0.00009009009 # 10m * 2 at the equator.
    pixel_size_y = 0.00009009009

    clipped_polygons_path = tile + '.shp'
    command = f'ogr2ogr -clipsrc {xmin} {ymin} {xmax} {ymax} {clipped_polygons_path} {ocean_polygons_path}'
    os.system(command)

    gdal.Rasterize(
        tile_tif,
        clipped_polygons_path,
        xRes=pixel_size_x,
        yRes=pixel_size_y, 
        burnValues=1,
        outputBounds=[xmin, ymin, xmax, ymax], 
        outputType=gdal.GDT_Byte
    )

    temp_files = [tile + '.dbf', tile + '.cpg', tile + '.prj', tile + '.shx']
    remove_temp_files(temp_files)


def extract_water(water_file, lat, lon, tile_width_deg, tile_height_deg, interior_tile_dir):
    """Rasterize a water tile from the processed global PBF file.

    Args:
        water_file: The path to the processed global PBF file.
        lat: The minimum latitude of the tile.
        lon: The minimum longitude of the tile.
        tile_width_deg: The desired width of the tile in degrees.
        tile_height_deg: The desired height of the tile in degrees.
    """

    tile = lat_lon_to_tile_string(lat, lon, is_worldcover=False, postfix='')
    tile_pbf = tile+ '.osm.pbf'
    tile_tif = interior_tile_dir + tile + '.tif'
    tile_shp = tile + '.shp'
    tile_geojson = tile + '.geojson'

    # Extract tile from the main pbf, then convert it to a tif.
    os.system(f'osmium extract -s smart -S tags=natural=water --bbox {lon},{lat},{lon+tile_width_deg},{lat+tile_height_deg} {water_file} -o {tile_pbf}')
    os.system(f'osmium export --geometry-types="polygon" {tile_pbf} -o {tile_geojson}')

    # Islands and Islets can be members of the water features, so they must be removed.
    water_gdf = gpd.read_file(tile_geojson, engine='pyogrio')
    try:
        water_gdf = water_gdf.drop(water_gdf[water_gdf['place'] == 'island'].index)
        water_gdf = water_gdf.drop(water_gdf[water_gdf['place'] == 'islet'].index)
    except:
        # When there are no islands to remove, an error will be occur, but we don't care about it.
        pass
    water_gdf.to_file(tile_shp, mode='w', engine='pyogrio')
    water_gdf = None

    xmin, xmax, ymin, ymax = lon, lon+tile_width_deg, lat, lat+tile_height_deg
    pixel_size_x = 0.00009009009 # 10m at the equator.
    pixel_size_y = 0.00009009009

    gdal.Rasterize(
        tile_tif,
        tile_shp,
        xRes=pixel_size_x, 
        yRes=pixel_size_y, 
        burnValues=1,
        outputBounds=[xmin, ymin, xmax, ymax], 
        outputType=gdal.GDT_Byte,
    )

    temp_files = [tile + '.dbf', tile + '.cpg', tile + '.prj', tile + '.shx', tile_shp, tile_pbf, tile_geojson]
    remove_temp_files(temp_files)


def merge_tiles(internal_tile_dir, ocean_tile_dir, finished_tile_dir, translate_to_cog: bool = False):
    """Merge the interior water tiles and ocean water tiles.
    
    Args:
        interior_tile_dir: The path to the directory containing the interior water tiles.
        ocean_tile_dir: The path to the directory containing the ocean water tiles.
        merged_tile_dir: The path to the directory containing the merged water tiles.
    """
    index = 0
    num_tiles = len([f for f in os.listdir(internal_tile_dir) if f.endswith('tif')])
    for filename in os.listdir(internal_tile_dir):
        if filename.endswith('.tif'):
            try:
                start_time = time.time()

                internal_tile = internal_tile_dir + filename
                external_tile = ocean_tile_dir + filename
                output_tile = finished_tile_dir + filename
                command = f'gdal_calc.py -A {internal_tile} -B {external_tile} --format GTiff --outfile {output_tile} --calc "logical_or(A, B)"'
                os.system(command)

                if translate_to_cog:
                    cogs_dir = finished_tile_dir + 'cogs/'
                    try:
                        os.mkdir(cogs_dir)
                    except FileExistsError as e:
                        pass
                    out_file = cogs_dir + filename
                    command = f'gdal_translate -ot Byte -of COG -co NUM_THREADS=all_cpus {output_tile} {out_file}'
                    os.system(command)

                end_time = time.time()
                total_time = end_time - start_time

                print(f'Elapsed Time: {total_time}(s)')
                print(f'Completed {index} of {num_tiles}')

                index += 1
            except Exception as e:
                print(f'Caught error while processing {filename}. Continuing anyways...\n{e}')


def setup_directories():
    """Setup the directories necessary for running the script."""
    dirs = [INTERIOR_TILE_DIR, OCEAN_TILE_DIR, FINISHED_TILE_DIR]
    for dir in dirs:
        try:
            os.mkdir(dir)
        except FileExistsError as e:
            print(f'{dir} already exists. Skipping...')


def main():

    parser = argparse.ArgumentParser(
        prog='generate_osm_tiles.py',
        description='Main script for creating a tiled watermask dataset from OSM data.'
    )

    parser.add_argument('--planet-file-path', help='The path to the global planet.pbf file.', default='planet.pbf')
    parser.add_argument('--ocean-polygons-path', help='The path to the global OSM ocean polygons.', default='water_polygons.shp')
    parser.add_argument('--lat-begin', help='The minimum latitude of the dataset.', default=-85)
    parser.add_argument('--lat-end', help='The maximum latitude of the dataset.', default=85)
    parser.add_argument('--lon-begin', help='The minimum longitude of the dataset.', default=-180)
    parser.add_argument('--lon-end', help='The maximum longitude of the dataset.', default=180)
    parser.add_argument('--tile-width', help='The desired width of the tile in degrees.', default=5)
    parser.add_argument('--tile-height', help='The desired height of the tile in degrees.', default=5)

    args = parser.parse_args()

    lat_begin = int(args.lat_begin)
    lat_end = int(args.lat_end)
    lon_begin = int(args.lon_begin)
    lon_end = int(args.lon_end)
    tile_width = int(args.tile_width)
    tile_height = int(args.tile_height)

    setup_directories()

    print('Extracting water from planet file...')
    processed_pbf_path = 'planet_processed.pbf'
    process_pbf(args.planet_file_path, processed_pbf_path)

    print('Processing tiles...')
    lat_range = range(lat_begin, lat_end, tile_height)
    lon_range = range(lon_begin, lon_end, tile_width)
    num_tiles = len(lat_range) * len(lon_range)
    index = 0
    for lat in lat_range:
        for lon in lon_range:
            tile_name = lat_lon_to_tile_string(lat, lon, is_worldcover=False)
            try:
                start_time = time.time()
                extract_water(processed_pbf_path, lat, lon, tile_width, tile_height, interior_tile_dir=INTERIOR_TILE_DIR)
                process_ocean_tiles(args.ocean_polygons_path, lat, lon, tile_width, tile_height, output_dir=OCEAN_TILE_DIR)
                end_time = time.time()
                total_time = end_time - start_time  #seconds
                print(f'Finished initial creation of {tile_name} in {total_time}(s). {index} of {num_tiles}')
                index += 1
            except Exception as e:
                print(f'Caught error while processing {tile_name}. Continuing anyways...\n{e}')

    print('Merging processed tiles...')                
    merge_tiles(INTERIOR_TILE_DIR, OCEAN_TILE_DIR, FINISHED_TILE_DIR, translate_to_cog=True)


if __name__ == '__main__':
    main()