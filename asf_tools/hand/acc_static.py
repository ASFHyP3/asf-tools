import argparse
import logging
#import sys
#import os
#import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
#from typing import Optional, Union

#import astropy.convolution
import fiona
#from osgeo import gdal
import numpy as np
import rasterio.crs
import rasterio.mask
from pysheds.sgrid import sGrid
import shapely
from shapely.geometry import GeometryCollection, shape

from asf_tools.composite import write_cog
from asf_tools.hand.upload_to_s3 import upload_file


log = logging.getLogger(__name__)


def calculate_acc(demfile):

    dem = rasterio.open(demfile)

    dem_array = dem.read(1)
    dem_affine = dem.meta["transform"]
    dem_crs = dem.crs

    nodata_fill_value = np.finfo(float).eps
    with NamedTemporaryFile() as temp_file:
        write_cog(temp_file.name, dem_array,
                  transform=dem_affine.to_gdal(), epsg_code=dem_crs.to_epsg(),
                  # Prevents PySheds from assuming using zero as the nodata value
                  nodata_value=nodata_fill_value)

        grid = sGrid.from_raster(str(temp_file.name))

        dem = grid.read_raster(str(temp_file.name))

    log.info('Fill pits in DEM')
    pit_filled_dem = grid.fill_pits(dem)

    log.info('Filling depressions')
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    del pit_filled_dem

    log.info('Resolving flats')
    inflated_dem = grid.resolve_flats(flooded_dem)
    del flooded_dem

    log.info('Obtaining flow direction')
    flow_dir = grid.flowdir(inflated_dem, apply_mask=True)

    log.info('Calculating flow accumulation')
    acc = grid.accumulation(flow_dir)

    return acc


def calc_mean_acc_with_basin(accfile, shpfile):
    # ds =  gdal.Open(file)

    # data = ds.GetRasterBand(1).ReadAsArray()

    acc = rasterio.open(accfile)

    with fiona.open(shpfile, "r") as shpf:
        shapes = [shapely.geometry.shape(feature["geometry"]) for feature in shpf]

    num = len(shapes)

    out = np.zeros(num, dtype=float)

    for k, p in enumerate(shapes):
        mask, tf, win = rasterio.mask.raster_geometry_mask(acc, [p], crop=True, pad=True, pad_width=1)

        out[k] = acc.read(window=win).mean()

    return out


def calc_mean_acc_with_subset(acc, num):

    #ds = rasterio.open(accfile, "r")

    #data = ds.read(1)

    rows, cols = acc.shape

    row_inc = int(rows / num)

    col_inc = int(cols / num)

    out = np.zeros((num, num), dtype=float)

    for i in range(num):
        for j in range(num):
            out[i, j] = acc[i * row_inc: (i + 1) * row_inc, j * col_inc: (j + 1) * col_inc].mean()

    return out


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('demfile', help=' tile dem file ')
    parser.add_argument('num', type=int, help=' number of subset is num*num')
    parser.add_argument('projname', type=str, help=' project name ')
    args = parser.parse_args()

    acc = calculate_acc(args.demfile)

    out = calc_mean_acc_with_subset(acc, args.num)

    # write the out array to file

    file = Path(args.demfile).name.split(".tif")[0]
    outfile = f"{file}_acc_{args.num}_{args.num}.npy"

    with open(outfile, "wb") as fid:
        np.save(fid, out)

    #upload outfile to s3

    upload_file(outfile, "jzhu4", f"{args.projname}")

    print("complete...")
