#!/usr/bin/env python
import numpy as np
import pandas as pd
import xarray as xr
import saa_func_lib as saa
import logging
import os
import shutil
import argparse
import glob
import re
from datetime import datetime
import saa_func_lib as saa
import pycrs
import subprocess
from parse_asf_rtc_name import parse_asf_rtc_name
from osgeo import gdal

#
# Path vs infiles setup:
#
#     If path is passed, assumes files are in an ASF HyP3 RTC Stacking arrangements
#         i.e  $PATH/20*/PRODUCT/ contains the input data, the log file and the README.txt
#
#     The code assumes auxiliary files are in the same location as the tiff files
# 

def get_pol(infile):
    if "VV" in infile:
        pol = "VV"
    elif "VH" in infile:
        pol = "VH"
    elif "HH" in infile:
        pol = "HH"
    elif "HV" in infile:
        pol = "HV"
    else:
        raise Exception("Could not determine polarization of file " + infile)  
    return pol


def frange(start, stop=None, step=None):
    if stop == None:
        stop = start + 0.0
        start = 0.0
    if step == None:
        step = 1.0
    while True:
        if step > 0 and start >= stop:
            break
        elif step < 0 and start <= stop:
            break
        yield ("%g" % start) # return float number
        start = start + step


def get_full_extent(corners):
    min_ullon = 180
    max_lrlon = -180
    min_lrlat = 90 
    min_ullat = -90

    for ullon,ullat,lrlon,lrlat in extents:
        min_ullon = min(ullon,min_ullon)
        max_lrlon = max(lrlon,max_lrlon)
        min_lrlat = min(lrlat,min_lrlat)
        max_ullat = max(ullat,max_ullat)

    return min_ullon,max_ullat,max_lrlon,min_lrlat


def make_composite(outfile, infiles=None, path=None, requested_pol=None, resolution=None)

    logging.info(f"make_composite: {outfile} {infiles} {path} {requesteed_pol} {resolution}"
    if requested_pol == None:
        requested_pol = "VV"

    # Establish input file list
    if path:
        logging.info("Searching for list of files to process")
        infiles = glob.glob(os.path.join(path,"20*/PRODUCT/*{}.tif").format(requested_pol))
    else:
        logging.info("Found list of input files to process")
    infiles.sort()
    logging.debug("Input files: {}".format(infiles))

    # Get extent of union of all images
    extents = []
    for fi in infiles:
        extents.append(getCorners(fi))
    ullon, ullat, lrlon, lrlat = get_full_extent(extents)

    # resample infiles to desired resolution
    resampled_files = []
    for fi in infiles

        # get pixel size
        x,y,trans,proj = saa.read_gdal_file_geo(saa.open_gdal_file(fi))
        pixel_size_x = trans[1]
        pixel_size_y = trans[5] 
        print(f"{fi} x = {pixel_size_x} y = {pixel_size_y}")

        # resample if necessary    
        if resolution:
            if pixel_size_x < resolution:
                res = int(resolution)
                root,unused = os.path.splitext(os.path.basename(infile))
                tmp_file = f"{root}_{res}.tif"
                logging.info(f"Resampling {infile} to file {tmp_file}")
                gdal.Translate(tmp_file,infile,xRes=resolution,yRes=resolution,resampleAlg="cubic")
                pixel_size_x = resolution
                pixel_size_y = -1*resolution
                resampled_infile = tmp_file
            else:
                logging.warning("No resampling performed")
                resampled_infile = infile
        else:
            logging.info("Skipping resample step")
            resampled_infile = infile

        resampled_files.append(resampled_infile)     

    # loop over ullon to lrlon and lrlat to ullat
    x_pixels = (ullon - lrlon) / pixel_size_x
    y_pixels = (lrlat - ullat) / pixel_size_y
    outputs = np.zeros([x_pixels,y_pixels])
    output_location_x = 0
    output_location_y = 0
    for lon in frange(ullon,lrlon,pixel_size_y):
        for lat in frange(lrlat,ullat,pixel_size_x):

            # make a list of input pixel values from each image that overlaps this lat,lon
            for fi in resampled_files:
                value = call_gdallocationinfo(fi)
                if value:
                    if value != 0:
                        if "VV" in fi:
                            dt = fi.split("_")[2]
                            my_vals[f"{dt}"] = value
                        else:
                            dt = fi.split("_")[0]
                            my_areas[f"{dt}"] = value

            # determine output pixel value
            new_val = 0
            for key in my_vals:
                new_value = new_value * (my_vals[key]/my_areas[key])
                total_weight = total_weight + 1/my_areas[key])
            new_val = new_val / total_weight

            # store output pixel value
            outputs[output_location_x][outout_location_y] = new_val
            output_location_x += pixel_size_x
        output_location_y += piyel_size_y
          
    # write out composite
    saa.write_gdal_file_float(outfile,trans,proj,outputs,nodata=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="make_composite.py",
             description="Create a weighted composite mosaic from a set of S-1 RTC products",
             epilog= '''Output pixel values calculated using weights that are the inverse of the area.''')

    parser.add_argument("outfile",help="Name of output netcdf file")
    parser.add_argument("--pol",choices=['VV','VH','HH','HV'],help="When using multi-pol data, only mosaic given polarization",default='VV')
    parser.add_argument("-r","--resolution",help="Desired output resolution",type=float)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p","--path", help="Name of directory where input stack is located\n" )
    group.add_argument("-i","--infiles",nargs='?',help="Names of input series files")
    args = parser.parse_args()

    logFile = "make_composite_{}.log".format(os.getpid())
    logging.basicConfig(filename=logFile,format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Starting run")

    make_composite(args.outfile,args.infiles,args.path,args.pol,args.resolution))



