import os
import time

import numpy as np

from osgeo import gdal


def lat_lon_to_tile_string(lat, lon, is_worldcover: bool = False, postfix: str ='.tif'):
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