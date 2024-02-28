import os
import time

import numpy as np

from osgeo import gdal


def lat_lon_to_tile_string(lat, lon, is_worldcover: bool = False, postfix: str ='.tif'):
    """Get the name of the tile with lower left corner (lat, lon).

    Args:
        lat: The minimum latitude of the tile.
        lon: The minimum longitude of the tile.
        is_worldcover: Wheter the tile is Worldcover or OSM.
        postfix: A postfix to append to the tile name to make it a filename.
    
    Returns:
        The name of the tile.
    """
    prefixes = ['N', 'S', 'E', 'W'] if is_worldcover else ['n', 's', 'e', 'w']
    if lat >= 0:
        lat_part = prefixes[0] + str(int(lat)).zfill(2)
    else:
        lat_part = prefixes[1] + str(int(np.abs(lat))).zfill(2)
    if lon >= 0:
        lon_part = prefixes[2] + str(int(lon)).zfill(3)
    else:
        lon_part = prefixes[3] + str(int(np.abs(lon))).zfill(3)
    return lat_part + lon_part + postfix


def merge_tiles(tiles, out_format, out_filename):
    """Merge tiles via buildvrt and translate.

    Args:
        tiles: The list of tiles to be merged.
        out_format: The format of the output image.
        out_filename: The name of the output COG.
    """
    vrt = 'merged.vrt'
    build_vrt_command = ' '.join(['gdalbuildvrt', vrt] + tiles)
    translate_command = ' '.join(['gdal_translate', '-of', out_format, vrt, out_filename])
    os.system(build_vrt_command)
    os.system(translate_command)
    os.remove(vrt)


def remove_temp_files(temp_files: list):
    """Remove each file in a list of files.
    
    Args:
        temp_files: The list of temporary files to remove.
    """
    for file in temp_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f'Caught {e} while removing temporary file: {file}. Skipping it...')


def setup_directories(dirs: list[str]):
    """Setup the directories necessary for running the script."""
    for dir in dirs:
        try:
            os.mkdir(dir)
        except FileExistsError as e:
            # Directories already exists.
            pass