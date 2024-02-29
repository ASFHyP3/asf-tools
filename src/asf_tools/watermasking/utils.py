import os
import subprocess

import numpy as np


def lat_lon_to_tile_string(lat, lon, is_worldcover: bool = False, postfix: str = '.tif'):
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


def merge_tiles(tiles, out_filename, out_format, compress=False):
    """Merge tiles via buildvrt and translate.

    Args:
        tiles: The list of tiles to be merged.
        out_format: The format of the output image.
        out_filename: The name of the output COG.
    """
    vrt = 'merged.vrt'
    build_vrt_command = ['gdalbuildvrt', vrt] + tiles
    if not compress:
        translate_command = ['gdal_translate', '-of', out_format, vrt, out_filename]
    else:
        translate_command = [
            'gdal_translate',
            '-of', out_format,
            '-co', 'COMPRESS=LZW',
            '-co', 'NUM_THREADS=all_cpus',
            vrt,
            out_filename
        ]
    subprocess.run(build_vrt_command)
    subprocess.run(translate_command)
    remove_temp_files([vrt])


def remove_temp_files(temp_files: list):
    """Remove each file in a list of files.

    Args:
        temp_files: The list of temporary files to remove.
    """
    for file in temp_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            print(f'Temp file {file} was not found, skipping removal...')


def setup_directories(dirs: list[str]):
    """Setup the directories necessary for running the script."""
    for directory in dirs:
        try:
            os.mkdir(directory)
        except FileExistsError:
            # Directories already exists.
            pass
