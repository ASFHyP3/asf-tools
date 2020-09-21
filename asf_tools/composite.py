#!/usr/bin/env python
import numpy as np
import logging
import os
import argparse
import glob
from datetime import datetime
from hyp3lib import saa_func_lib as saa
from osgeo import gdal
from subprocess import Popen, PIPE


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
    min_ulx = 50000000
    max_lrx = 0 
    max_uly = 0
    min_lry = 50000000

    for fi,ulx,lrx,lry,uly in corners:
        print(f"{ulx,uly} {lrx,lry}")
        min_ulx = min(ulx,min_ulx)
        max_uly = max(uly,max_uly)
        max_lrx = max(lrx,max_lrx)
        min_lry = min(lry,min_lry)

    print(f"Return is upper left: {min_ulx,max_uly}; lower right: {max_lrx,min_lry}")
    return min_ulx,max_lrx,max_uly,min_lry


def call_gdallocationinfo(fi,y,x):
    cmd = ['gdallocationinfo', '-geoloc', '-valonly', fi, x, y]
    with Popen(cmd, stdout=PIPE) as proc:
        val = proc.stdout.read()
        try:
            val = float(val)
        except: 
            val = None 
    return (val)


def make_composite(outfile, infiles=None, path=None, requested_pol=None, resolution=None):

    logging.info(f"make_composite: {outfile} {infiles} {path} {requested_pol} {resolution}")
    if requested_pol == None:
        requested_pol = "VV"

    # Establish input file list
    if path:
        logging.info("Searching for list of files to process")
        infiles = glob.glob(os.path.join(path,"20*/PRODUCT/*{}.tif").format(requested_pol))
    else:
        logging.info("Found list of input files to process")
    infiles.sort()
    logging.debug(f"Input files: {infiles}")

    # resample infiles to desired resolution
    if resolution:
        resampled_files = []
        for fi in infiles:
            x,y,trans,proj = saa.read_gdal_file_geo(saa.open_gdal_file(fi))
            pixel_size_x = trans[1]
            pixel_size_y = trans[5] 
            print(f"{fi} x = {pixel_size_x} y = {pixel_size_y}")

            if pixel_size_x < resolution:
                res = int(resolution)
                root,unused = os.path.splitext(os.path.basename(fi))
                tmp_file = f"{root}_{res}.tif"
                logging.info(f"Resampling {fi} to file {tmp_file}")
                gdal.Translate(tmp_file, fi, xRes=resolution, yRes=resolution, resampleAlg="cubic")
                pixel_size_x = resolution
                pixel_size_y = -1 * resolution
                resampled_infile = tmp_file
            else:
                logging.warning("No resampling performed")
                resampled_infile = fi
            resampled_files.append(resampled_infile)     
    else:
        logging.info("Skipping resample step")
        x,y,trans,proj = saa.read_gdal_file_geo(saa.open_gdal_file(infiles[0]))
        pixel_size_x = trans[1]
        pixel_size_y = trans[5] 
        print(f"{infiles[0]} x = {pixel_size_x} y = {pixel_size_y}")
        resampled_files = infiles

    # Get extent of union of all images
    extents = []
    for fi in resampled_files:
        ulx,lrx,lry,uly = saa.getCorners(fi)
        extents.append([fi,ulx,lrx,lry,uly])
    ulx, lrx, uly, lry = get_full_extent(extents)

    print(f"Full extent of mosaic is {ulx,uly} to {lrx,lry}")
   
    x_pixels = abs(int((ulx - lrx) / pixel_size_x))
    y_pixels = abs(int((lry - uly) / pixel_size_y))

    print(f"Output size is {x_pixels} samples by {y_pixels} lines")

    outputs = np.zeros((y_pixels,x_pixels))
    weights = np.zeros((y_pixels,x_pixels))
    counts = np.zeros((y_pixels,x_pixels),dtype=np.int8)
    logging.info("Calculating output values")

    for fi,x_max,x_min,y_max,y_min in extents:
        if "VV" in fi:
            print(f"Processing file {fi}")
            print(f"File covers {x_max,y_min} to {x_min,y_max}")

            print("Reading values")
            x_size, y_size, trans, proj, areas = saa.read_gdal_file(saa.open_gdal_file(fi.replace("_flat_VV","_area_map")))

            # Set zero area to a large number to
            #  - protect against Nans in outputs
            #  - not skew the weights
            areas[areas == 0] = 10000000

            print("Reading areas")
            x_size, y_size, trans, proj, values = saa.read_gdal_file(saa.open_gdal_file(fi))

            out_loc_x = (x_max - ulx) / pixel_size_x
            out_loc_y = (y_min - uly) / pixel_size_y
            end_loc_x = out_loc_x + x_size
            end_loc_y = out_loc_y + y_size

            print(f"Placing values in output grid at {int(out_loc_x)}:{int(end_loc_x)} and {int(out_loc_y)}:{int(end_loc_y)}")

            outputs[int(out_loc_y):int(end_loc_y), int(out_loc_x):int(end_loc_x)] += values * 1.0/areas
            weights[int(out_loc_y):int(end_loc_y), int(out_loc_x):int(end_loc_x)] += 1.0/areas 
            counts[int(out_loc_y):int(end_loc_y), int(out_loc_x):int(end_loc_x)] += 1
             
            # write out composite
            tmpfile = f"composite_{fi}"
            saa.write_gdal_file_float(tmpfile,trans,proj,outputs,nodata=0)

    outputs /= weights            

    # write out composite
    saa.write_gdal_file_float(outfile,trans,proj,outputs,nodata=0)
    saa.write_gdal_file("counts.tif",trans,proj,counts.astype(np.int16))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="make_composite.py",
             description="Create a weighted composite mosaic from a set of S-1 RTC products",
             epilog= '''Output pixel values calculated using weights that are the inverse of the area.''')

    parser.add_argument("outfile",help="Name of output weighted mosaic geotiff file")
    parser.add_argument("--pol",choices=['VV','VH','HH','HV'],help="When using multi-pol data, only mosaic given polarization",default='VV')
    parser.add_argument("-r","--resolution",help="Desired output resolution",type=float)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p","--path", help="Name of directory where input stack is located\n" )
    group.add_argument("-i","--infiles",nargs='*',help="Names of input series files")
    args = parser.parse_args()

    logFile = "make_composite_{}.log".format(os.getpid())
    logging.basicConfig(filename=logFile,format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Starting run")

    make_composite(args.outfile,args.infiles,args.path,args.pol,args.resolution)



