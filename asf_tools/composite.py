#!/usr/bin/env python
import numpy as np
import logging
import os
import re
import argparse
import glob
from datetime import datetime
from hyp3lib import saa_func_lib as saa
from osgeo import gdal
from subprocess import Popen, PIPE
from osgeo.gdalconst import GRIORA_Cubic

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


def get_max_pixel_size(files):
    pix_size = -999
    for fi in files:
        (x1, y1, t1, p1) = saa.read_gdal_file_geo(saa.open_gdal_file(fi))
        tmp = t1[1]
        pix_size = max(pix_size, tmp)

    if pix_size == -999:
        Exception("No valid pixel sizes found")
    return pix_size


def get_hemisphere(fi):
    hemi = None
    dst = gdal.Open(fi)
    p1 = dst.GetProjection()
    ptr = p1.find("UTM zone ")
    if ptr != -1:
        (zone, hemi) = [t(s) for t, s in zip((int, str), re.search('(\d+)(.)', p1[ptr:]).groups())]
    return hemi


def get_zone_from_proj(fi):
    zone = None
    dst = gdal.Open(fi)
    p1 = dst.GetProjection()
    ptr = p1.find("UTM zone ")
    if ptr != -1:
        (zone, hemi) = [t(s) for t, s in zip((int, str), re.search("(\d+)(.)", p1[ptr:]).groups())]
    return zone


def parse_zones(files):
    zones = []
    for fi in files:
        zone = get_zone_from_proj(fi)
        if zone:
            zones.append(zone)
    return np.asarray(zones, dtype=np.int8)


def reproject_to_median_utm(files,resolution=None):

    if len(files) < 2:
        return None 

    # Set the pixel size
    if resolution:
        pix_size = resolution
        print(f"Changing pixel size to {pix_size}")
    else:
        pix_size = get_max_pixel_size(files)
        print(f"Using maximum pixel size {pix_size}")

    # Get the median UTM zone and hemisphere
    home_zone = np.median(parse_zones(files))
    print(f"Home zone is {home_zone}")
    hemi = get_hemisphere(files[0])
    print(f"Hemisphere is {hemi}")

    # Reproject files as needed
    print("Checking projections")
    new_files = []
    for fi in files:
        my_zone = get_zone_from_proj(fi)
        name = fi.replace(".tif", "_reproj.tif")
        afi = fi.replace("_flat_VV.tif","_area_map.tif")
        aname = fi.replace("_flat_VV.tif","_area_map_reproj.tif")
        if my_zone != home_zone:
            print(f"Reprojecting {fi} to {name}")
            if hemi == "N":
                proj = ('EPSG:326%02d' % int(home_zone))
            else:
                proj = ('EPSG:327%02d' % int(home_zone))
            gdal.Warp(name, fi, dstSRS=proj, xRes=pix_size, yRes=pix_size, targetAlignedPixels=True,resampleAlg=GRIORA_Cubic)
            gdal.Warp(aname, afi, dstSRS=proj, xRes=pix_size, yRes=pix_size, targetAlignedPixels=True,resampleAlg=GRIORA_Cubic)
        else:
            # May need to reproject to desired resolution
            x,y,trans,proj = saa.read_gdal_file_geo(saa.open_gdal_file(fi))
            if x < pix_size:
                print(f"Changing resolution of {fi} to {pix_size}")
                gdal.Warp(name, fi, xRes=pix_size, yRes=pix_size, targetAlignedPixels=True,resampleAlg=GRIORA_Cubic)
                gdal.Warp(aname, afi, xRes=pix_size, yRes=pix_size, targetAlignedPixels=True,resampleAlg=GRIORA_Cubic)
            else:
                print(f"Linking {fi} to {name}")
                os.symlink(fi,name)
                os.symlink(afi,aname)
        new_files.append(name)

    print("All files completed")
    return new_files


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

    # resample infiles to maximum resolution & common UTM zone
    resampled_files = reproject_to_median_utm(infiles,resolution) 
    if len(resampled_files) == 0:
        exception("Unable to resample files")
        exit -1

    # Get pixel size
    x,y,trans,proj = saa.read_gdal_file_geo(saa.open_gdal_file(resampled_files[0]))
    pixel_size_x = trans[1]
    pixel_size_y = trans[5] 
    print(f"{resampled_files[0]} x = {pixel_size_x} y = {pixel_size_y}")

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

            print("Reading areas")
            x_size, y_size, trans, proj, areas = saa.read_gdal_file(saa.open_gdal_file(fi.replace("_flat_VV_reproj","_area_map_reproj")))

            # Set zero area to a large number to
            #  - protect against Nans in outputs
            #  - not skew the weights
            areas[areas == 0] = 10000000

            print("Reading values")
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



