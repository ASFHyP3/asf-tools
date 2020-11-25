"""Create S1 SAR Composite Mosaic using inverse area weighting ala David Small.

   Path vs infiles:
     If path is passed, code assumes files are in an ASF HyP3 RTC Stacking arrangement.
     i.e  {path}/20*/PRODUCT/ contains the input RTC data and the area maps or
          {path}/S1?_IW_*RTC*/ contains the input RTC data and the area maps


"""

import argparse
import glob
import logging
import os

import numpy as np
from hyp3lib import saa_func_lib as saa
from osgeo import gdal, osr


def get_epsg_code(file_name):
    info = gdal.Info(file_name, format='json')
    proj = osr.SpatialReference(info['coordinateSystem']['wkt'])
    epsg_code = proj.GetAttrValue('AUTHORITY', 1)
    return f'EPSG:{epsg_code}'


def get_target_epsg_code(files):
    epsg_codes = [get_epsg_code(f) for f in files]
    zones = [int(epsg_code[-2:]) for epsg_code in epsg_codes]
    target_zone = int(np.median(zones))
    target_epsg_code = epsg_codes[0][:-2] + str(target_zone)
    return target_epsg_code


def frange(start, stop=None, step=None):
    """Return a floating point number ranging from start to stop, adding step"""
    if not stop:
        stop = start + 0.0
        start = 0.0
    if not step:
        step = 1.0
    while True:
        if step > 0 and start >= stop:
            break
        elif step < 0 and start <= stop:
            break
        yield ("%g" % start)  # return float number
        start = start + step


def get_full_extent(corners):
    """"Calculate the union of corners"""
    min_ulx = 50000000
    max_lrx = 0
    max_uly = 0
    min_lry = 50000000

    for fi, ulx, lrx, lry, uly in corners:
        logging.debug(f"{ulx,uly} {lrx,lry}")
        min_ulx = min(ulx, min_ulx)
        max_uly = max(uly, max_uly)
        max_lrx = max(lrx, max_lrx)
        min_lry = min(lry, min_lry)

    logging.debug(f"Return is upper left: {min_ulx,max_uly}; lower right: {max_lrx,min_lry}")
    return min_ulx, max_lrx, max_uly, min_lry


def get_max_pixel_size(files):
    """Find maximum pixel size of given files"""
    pix_size = -999
    for fi in files:
        (x1, y1, t1, p1) = saa.read_gdal_file_geo(saa.open_gdal_file(fi))
        tmp = t1[1]
        pix_size = max(pix_size, tmp)

    if pix_size == -999:
        raise Exception("No valid pixel sizes found")
    return pix_size


def reproject_to_median_utm(files, pol, resolution=None):
    """Reproject a bunch of UTM geotiffs to the median UTM zone.
       Use either the given resolution or the largest resolution in the stack"""

    if len(files) < 2:
        return None

    # Set the pixel size
    if resolution:
        pix_size = resolution
        logging.info(f"Changing pixel size to {pix_size}")
    else:
        pix_size = get_max_pixel_size(files)
        logging.info(f"Using maximum pixel size {pix_size}")

    # Get the median EPSG code of all input files
    target_epsg_code = get_target_epsg_code(files)
    logging.info(f"Target EPSG code is {target_epsg_code}")

    # Reproject files as needed
    logging.info("Checking projections")
    new_files = []
    for fi in files:
        fi_name = os.path.split(fi)[1]
        my_epsg_code = get_epsg_code(fi)
        name = fi_name.replace(".tif", "_reproj.tif")
        afi = fi.replace(f"_{pol}.tif", "_area.tif")
        aname = fi_name.replace(f"_{pol}.tif", "_area_reproj.tif")
        if not os.path.isfile(name):
            x, y, trans, proj = saa.read_gdal_file_geo(saa.open_gdal_file(fi))
            if my_epsg_code != target_epsg_code:
                logging.info(f"Reprojecting {fi} to {name}")
                gdal.Warp(name, fi, dstSRS=target_epsg_code, xRes=pix_size, yRes=pix_size, targetAlignedPixels=True)
                gdal.Warp(aname, afi, dstSRS=target_epsg_code, xRes=pix_size, yRes=pix_size, targetAlignedPixels=True)
                new_files.append(name)
            elif x < pix_size:
                # Need to reproject to desired resolution
                logging.info(f"Changing resolution of {fi} to {pix_size}")
                gdal.Warp(name, fi, xRes=pix_size, yRes=pix_size, targetAlignedPixels=True)
                gdal.Warp(aname, afi, xRes=pix_size, yRes=pix_size, targetAlignedPixels=True)
                new_files.append(name)
            else:
                logging.info(f"No reprojection needed; Linking {fi} to {name}")
                os.symlink(fi, name)
                os.symlink(afi, aname)
                new_files.append(name)
        else:
            logging.info(f"Found previous reproj file {fi} - taking no action")
            new_files.append(fi)

    logging.info("All files completed")
    return new_files


def make_composite(outfile, infiles=None, path=None, pol=None, resolution=None):

    '''Create a composite mosaic of infiles using inverse area weighting to adjust backscatter'''

    logging.info(f"make_composite: {outfile} {infiles} {path} {pol} {resolution}")
    if pol is None:
        pol = "VV"

    # Establish input file list
    if path:
        logging.info("Searching for list of files to process")

        # New format directory names
        infiles_new = glob.glob(os.path.join(path, f"20*/S1?_IW_*RTC*/*{pol}.tif"))

        # Old format diretory names
        infiles_old = glob.glob(os.path.join(path, f"20*/PRODUCT/*{pol}.tif"))

        infiles = infiles_new
        infiles.extend(infiles_old)
        cnt = len(infiles)
        logging.info(f"Found {cnt} files to process")
    else:
        cnt = len(infiles)
        logging.info(f"Found list of {cnt} input files to process")
    infiles.sort()
    logging.debug(f"Input files: {infiles}")

    # resample infiles to maximum resolution & common UTM zone
    resampled_files = reproject_to_median_utm(infiles, pol, resolution=resolution)
    if len(resampled_files) == 0:
        Exception("Unable to resample files")

    # Get pixel size
    x, y, trans, proj = saa.read_gdal_file_geo(saa.open_gdal_file(resampled_files[0]))
    pixel_size_x = trans[1]
    pixel_size_y = trans[5]
    logging.info(f"{resampled_files[0]} x = {pixel_size_x} y = {pixel_size_y}")

    # Get extent of union of all images
    extents = []
    for fi in resampled_files:
        ulx, lrx, lry, uly = saa.getCorners(fi)
        extents.append([fi, ulx, lrx, lry, uly])
    ulx, lrx, uly, lry = get_full_extent(extents)

    logging.info(f"Full extent of mosaic is {ulx,uly} to {lrx,lry}")

    x_pixels = abs(int((ulx - lrx) / pixel_size_x))
    y_pixels = abs(int((lry - uly) / pixel_size_y))

    logging.info(f"Output size is {x_pixels} samples by {y_pixels} lines")

    outputs = np.zeros((y_pixels, x_pixels))
    weights = np.zeros((y_pixels, x_pixels))
    counts = np.zeros((y_pixels, x_pixels), dtype=np.int8)
    logging.info("Calculating output values")

    for fi, x_max, x_min, y_max, y_min in extents:
        if pol in fi:
            logging.info(f"Processing file {fi}")
            logging.info(f"File covers {x_max,y_min} to {x_min,y_max}")

            logging.info("Reading areas")
            x_size, y_size, trans, proj, areas = saa.read_gdal_file(saa.open_gdal_file(fi.replace(f"_{pol}_reproj",
                                                                                                  "_area_reproj")))

            logging.info("Reading values")
            x_size, y_size, trans, proj, values = saa.read_gdal_file(saa.open_gdal_file(fi))

            out_loc_x = (x_max - ulx) / pixel_size_x
            out_loc_y = (y_min - uly) / pixel_size_y
            end_loc_x = out_loc_x + x_size
            end_loc_y = out_loc_y + y_size

            logging.info(f"Placing values in output grid at {int(out_loc_x)}:{int(end_loc_x)} "
                         f"and {int(out_loc_y)}:{int(end_loc_y)}")

            temp = 1.0/areas
            temp[values == 0] = 0
            mask = np.ones((y_size, x_size), dtype=np.uint8)
            mask[values == 0] = 0

            outputs[int(out_loc_y):int(end_loc_y), int(out_loc_x):int(end_loc_x)] += values * temp
            weights[int(out_loc_y):int(end_loc_y), int(out_loc_x):int(end_loc_x)] += temp
            counts[int(out_loc_y):int(end_loc_y), int(out_loc_x):int(end_loc_x)] += mask

    # Divide by the total weight applied
    outputs /= weights

    # write out composite
    logging.info("Writing output files")

    easting = ulx
    weres = pixel_size_x
    werotation = trans[2]
    northing = uly
    nsrotation = trans[4]
    nsres = pixel_size_y

    trans = (easting, weres, werotation, northing, nsrotation, nsres)

    saa.write_gdal_file_float(outfile, trans, proj, outputs, nodata=0)
    saa.write_gdal_file("counts.tif", trans, proj, counts.astype(np.int16))

    logging.info("Program successfully completed")


def main():
    parser = argparse.ArgumentParser(
        prog="make_composite.py",
        description="Create a weighted composite mosaic from a set of S-1 RTC products",
        epilog="Output pixel values calculated using weights that are the inverse of the area."
    )

    parser.add_argument("outfile", help="Name of output weighted mosaic geotiff file")
    parser.add_argument("--pol", choices=['VV', 'VH', 'HH', 'HV'], default='VV',
                        help="When using multi-pol data, only mosaic given polarization")
    parser.add_argument("-r", "--resolution", type=float, help="Desired output resolution")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-p", "--path", help="Name of directory where input stack is located")
    group.add_argument("-i", "--infiles", nargs='*', help="Names of input series files")

    args = parser.parse_args()

    logFile = "make_composite_{}.log".format(os.getpid())
    logging.basicConfig(filename=logFile, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Starting run")

    make_composite(args.outfile, args.infiles, args.path, args.pol, args.resolution)
