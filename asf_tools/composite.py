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

def get_dataset(infiles,path,requested_pol):
    if path:
        logging.info("Searching for list of files to process")
        infiles = glob.glob(os.path.join(path,"20*/PRODUCT/*{}.tif").format(requested_pol))
    else:
        logging.info("Found list of input files to process")
    infiles.sort()
    logging.debug("Input files: {}".format(infiles))

    dataset = []
    all_time = []
    for infile in infiles:
        ulx_coord,lrx_coord,lry_coord,uly_coord = saa.getCorners(infile)
        x_coord = [ulx_coord,lrx_coord]
        y_coord = [uly_coord,lry_coord]
        image_dts = os.path.basename(infile)[12:27]
        granule = get_granule(path,image_dts,infile)
        proc_dt = get_proc_dt(path,image_dts,infile,granule)
        image_dt = datetime(int(image_dts[0:4]),int(image_dts[4:6]),int(image_dts[6:8]),
                           int(image_dts[9:11]),int(image_dts[11:13]),int(image_dts[13:15]),0)
        dataset.append([image_dt,proc_dt,infile,x_coord,y_coord,granule])
        all_time.append(image_dt)

    # read in the metadata for the last file
    x,y,trans,proj = saa.read_gdal_file_geo(saa.open_gdal_file(infiles[-0]))

    # parse projection string
    cfg_p = parse_proj_crs(proj) 
   
    dataset.sort(key=lambda row: row[0])

    # calculate milliseconds since first image in stack    
    time_0_str = dataset[0][0]
    time_0 = np.datetime64(time_0_str,'ms').astype(np.float)

    new_times = [] 
    for i in range(0,len(all_time)):
        milliseconds_since = (np.datetime64(all_time[i],'ms').astype(np.float)-time_0)
        new_times.append(milliseconds_since)

    logging.info("Sorted metadata dataset:")
    logging.info("  {}".format(dataset))

    return(dataset,cfg_p,new_times,time_0_str)


def get_granule(path,image_dts,infile): 
    lines,full_file_name = read_log_file(path,image_dts,infile)
    search_string = "S1[AB]_.._...._1S[DS][HV]_"+"{}".format(image_dts)+"_\d{8}T\d{6}_\d+_([0-9A-Fa-f]+)_([0-9A-Fa-f]+)"
    granule = re.search(search_string,lines)
    if not granule:
        raise Exception("ERROR: No granule name found in {}".format(full_file_name))
    granule = granule.group(0)
    logging.info("Found granule {}".format(granule))
    return granule


def get_proc_dt(path,image_dts,infile,granule): 
    lines,full_file_name = read_log_file(path,image_dts,infile)

    date_string = re.search("\d\d/\d\d/\d\d\d\d \d\d:\d\d:\d\d [A,P]M - INFO -\s*Input name\s*: {}".format(granule),lines)
    if not date_string:
        raise Exception("ERROR: No date_string found in {}".format(full_file_name))
    date_string = date_string.group(0)
    proc_date = re.search("\d\d/\d\d/\d{4} \d\d:\d\d:\d\d [A,P]M",date_string)
    if not proc_date:
        raise Exception("ERROR: No processing date found in {}".format(date_string))
    proc_date = proc_date.group(0)
    date, time, period = proc_date.split()
    hour, minute, second =  time.split(":")
    if "PM" in period and int(hour) != 12:
        hour = int(hour) + 12
    proc_date = datetime(int(date[6:10]),int(date[0:2]),int(date[3:5]),int(hour),int(minute),int(second))
    logging.info("Processing_date {}".format(proc_date))
    return proc_date


def read_log_file(path,image_dts,infile):
    if path:
        full_path = os.path.join(path,image_dts)
        full_path = os.path.join(full_path,"PRODUCT")
    else:
        full_path = os.path.dirname(infile)
    pol = get_pol(infile) 
    file_name = infile.replace("_{}.tif".format(pol),".log")
    file_name = os.path.basename(file_name)
    full_file_name = os.path.join(full_path,file_name)
    if not os.path.exists(full_file_name):
        raise Exception ("ERROR: Unable to find file {}".format(full_file_name))
    logging.debug("Reading file {}".format(full_file_name))
    with open(full_file_name,"r") as f:
        lines = f.read()
    return lines,full_file_name

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

def parse_proj_crs(proj_crs):

    print("Entering parse_proj_crs")
    print("proj_crs = {}".format(proj_crs))

    crs = pycrs.parse.from_ogc_wkt(proj_crs)

    print(f"crs = {crs}")
    print(f"proj = {crs.proj}")


    print(f"crs.geogcs.datum.name.ogc_wkt = {crs.geogcs.datum.name.ogc_wkt}")
    cfg_p = {}
    cfg_p['grid_mapping_name'] = crs.name
    cfg_p['crs_wkt'] = crs.proj.name.ogc_wkt.lower()

    # Is there a better way to do this? 
    for p in crs.params:
        if isinstance(p,pycrs.elements.parameters.LatitudeOrigin):
            cfg_p['latitude_of_projection_origin'] = p.value
        if isinstance(p,pycrs.elements.parameters.CentralMeridian):
            cfg_p['longitude_of_central_meridian'] = p.value
        if isinstance(p,pycrs.elements.parameters.FalseEasting):
            cfg_p['false_easting'] = p.value
        if isinstance(p,pycrs.elements.parameters.FalseNorthing):
            cfg_p['false_northing'] = p.value
        if isinstance(p,pycrs.elements.parameters.ScalingFactor):
            cfg_p['scale_factor_at_centeral_meridian'] = p.value

    cfg_p['projected_coordinate_system_name'] = crs.name
    cfg_p['geographic_coordinate_system_name'] = crs.geogcs.name
    cfg_p['horizontal_datum_name'] = crs.geogcs.datum.name.ogc_wkt
    cfg_p['reference_ellipsoid_name'] = crs.geogcs.datum.ellips.name.ogc_wkt
    cfg_p['semi_major_axis'] = crs.geogcs.datum.ellips.semimaj_ax.value
    cfg_p['inverse_flattening'] = crs.geogcs.datum.ellips.inv_flat.value
    cfg_p['longitude_of_prime_meridian'] = crs.geogcs.prime_mer.value
    cfg_p['units'] = crs.unit.unitname.ogc_wkt
    cfg_p['projection_x_coordinate'] = "x"
    cfg_p['projection_y_coordinate'] = "y"

    return cfg_p


def fill_cfg(prod_type):
    logging.info("Adding metadata")

    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "config"))
    ver_file = os.path.join(cfg_dir,"version.txt")
    if not os.path.exists(ver_file):
        logging.error("Unable to find stack software version file.  Looking for {}".format(ver_file))  
    try:
        f = open(ver_file,"r")
        for line in f:
            ss_ver = line.strip()
    except:
        logging.warning("Unable to open file {}.  Using UNKNOWN for stack software version.".format(ver_file))
        ss_ver = "UNKNOWN"

    cfg = {} 
    cfg['long_name'] = "SAR RTC Time Series"
    cfg['description'] = "This is RTC SAR data"
    cfg['product_type'] = prod_type
    cfg['date_processed'] = datetime.now().strftime("%Y%m%dT%H%M%S")
    cfg['stack_software_version'] = ss_ver
    cfg['institution'] = "Alaska Sattelite Facility (ASF)"
    cfg['source'] = "SAR observation"
    cfg['Conventions'] = "CF-1.8"
    cfg['feature_type'] = "timeSeries"
    cfg['title'] = 'SAR RTC Time Series'          # what's in the dataset
    cfg['history'] = ''                           # audit trail of process chain with time stamps
    cfg['references'] = 'asf.alaska.edu'          # published or web-based refernces
    cfg['comment'] = 'This is an early prototype. Contains modified Copernicus Sentinel data processed by ESA and ASF'
                                                  # misc info about the data
    cfg_rf = {}
    cfg_rw = {}
    cfg_rf['radiation_frequency_unit'] = "GHz"
    cfg_rw['radiation_wavelength_unit'] = "m"

    cfg_x = {}
    cfg_x['axis'] = "X"
    cfg_x['units'] = "m"
    cfg_x['standard_name']="projection_x_coordinate"
    cfg_x['long_name'] = "Easting"

    cfg_y = {}
    cfg_y['axis'] = "Y"
    cfg_y['units'] = "m"
    cfg_y['standard_name']="projection_y_coordinate"
    cfg_y['long_name'] = "Northing"
    
    return cfg,cfg_x,cfg_y,cfg_rf,cfg_rw


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


def get_science_code(infile):
    readme = os.path.join(os.path.dirname(infile),"README.txt")
    with open(readme,'r') as f:
        lines = f.readlines()

    sc_name =  "UNKNOWN"  
    for line in lines:
        if "RTC" in line and "GAMMA" in line:
            sc_name = "RTC GAMMA"
        elif "RTC" in line and "S1TBX" in line:
            sc_name = "RTC S1TBX"
    
        if "HYP3" in line and "software version" in line:
            obj = re.search("\d+\.\d+\.*\d*",line)
            sc_ver = obj.group(0)

        if "release" in line and "{}".format(sc_name[0]) in line:
           obj = re.search("\d{8}",line)
           pk_ver = obj.group(0)

    return(sc_name,sc_ver,pk_ver)


def create_dB(fi,input_type):
    (x,y,trans,proj,data) = saa.read_gdal_file(saa.open_gdal_file(fi))
    if "amp" in input_type:
        pwrdata = data*data
        dBdata = 10 * np.log(pwrdata)
    elif "power" in input_type:
        dBdata = 10 * np.log(data)
    outfile = fi.replace('.tif','_dB.tif')
    saa.write_gdal_file_float(outfile,trans,proj,dBdata,nodata=0)
    return(outfile)

def pwr2amp(fi):
    x,y,trans,proj,data = saa.read_gdal_file(saa.open_gdal_file(fi))
    ampdata = np.sqrt(data)
    outfile = fi.replace(".tif","_amp.tif")
    saa.write_gdal_file_float(outfile,trans,proj,ampdata,nodata=0)
    return(outfile)

def amp2pwr(fi):
    x,y,trans,proj,data = saa.read_gdal_file(saa.open_gdal_file(fi))
    pwrdata = data * data 
    outfile = fi.replace(".tif","_pwr.tif")
    saa.write_gdal_file_float(outfile,trans,proj,pwrdata,nodata=0)
    return(outfile)

def scaleFile(infile,otype,prod_type):
    # get the scaling type 
    if "RTC" in prod_type:
        scene = parse_asf_rtc_name(os.path.basename(infile))
    else: 
        raise Exception("Unknown product type {}".format(prod_type))

    # rescale if necesasry
    if scene['scale'] == "amp" and otype == "power":
         logging.info("Converting amplitude to power")
         scaled_file = amp2pwr(infile)
    elif scene['scale'] == "power" and otype == "amp":
         logging.info("Converting power to amplitude")
         scaled_file = pwr2amp(infile)
    elif otype == "dB":
         logging.info("Converting {} to dB".format(scene['scale']))
         scaled_file = create_dB(infile,scene['scale'])
    else:
         scaled_file = infile

    return(scaled_file)


def make_composite(outfile,infiles=None,path=None,otype=None,requested_pol=None,resolution=None):

    logging.info("make_composite: {} {} {} {} {} {}".format(prod_type,outfile,infiles,path,otype,requested_pol))
    if requested_pol == None:
        requested_pol = "VV"
    dataset,cfg_p,all_time,time_0_str = get_dataset(infiles,path,requested_pol)
    cfg,cfg_x,cfg_y,cfg_rf,cfg_rw = fill_cfg("SAR_RTC_timeSeries")
    logging.info("Config is {}".format(cfg))

    xarr3d = None
    started = False 

    logging.info("============================================================================")
    metadata = []

    for image_dt,proc_dt,infile,x_extent,y_extent,granule in dataset: 
        logging.info("Adding layer {}".format(infile)) 

        # get pixel size
        x,y,trans,proj = saa.read_gdal_file_geo(saa.open_gdal_file(infile))
        pix_x = trans[1]
        pix_y = trans[5] 
        print(f"x = {pix_x} y = {pix_y}")

        # resample if necessary    
        if resolution:
            if pix_x < resolution:
                res = int(resolution)
                root,unused = os.path.splitext(os.path.basename(infile))
                tmp_file = f"{root}_{res}.tif"
                logging.info(f"Resampling {infile} to file {tmp_file}")
                gdal.Translate(tmp_file,infile,xRes=resolution,yRes=resolution,resampleAlg="cubic")
                pix_x = resolution
                pix_y = -1*resolution
                resampled_infile = tmp_file
            elif pix_x >= resolution:
                logging.warning(f"Desired output resolution less than original  ({resolution} vs {pix_x})")
                logging.warning("No resampling performed")
                resampled_infile = infile
        else:
            logging.info("Skipping resample step")
            resampled_infile = infile

        # Convert to power, amp or dB 
        scaled_file = scaleFile(resampled_infile,otype,prod_type)
        
        # read in the data file
        x,y,trans,proj,data = saa.read_gdal_file(saa.open_gdal_file(scaled_file))
        pix_x = trans[1]
        pix_y = trans[5] 
        
#        print(f"x = {x} y = {y} [from image]")
#
#        # fill x_coord, y_coord with ranges  
#        print(f"x_extent {x_extent}")
#        print(f"y_extent {y_extent}")
# 
        ulx_coord,lrx_coord,lry_coord,uly_coord = saa.getCorners(scaled_file)
        x_extent = [ulx_coord,lrx_coord]
        y_extent = [uly_coord,lry_coord]

        x_coord = np.arange(x_extent[0],x_extent[1],pix_x)
        y_coord = np.arange(y_extent[0],y_extent[1],pix_y)
 
#        print("Before size check:")
#        print(f"x_coord len = {len(x_coord)}; x_coords = {x_coord}")
#        print(f"y_coord len = {len(y_coord)}; y_coords = {y_coord}")
# 
        if len(x_coord) > x:
#            print(f"coordinate length1 {len(x_coord)}")
#            print(f"image length {x}")
#          
            x_coord = x_coord[0:x]
#
        if len(y_coord) > y:
            y_coord = y_coord[0:y]
#
#        print(f"After length check:")
#        print(f"x_coord len = {len(x_coord)}; x_coords = {x_coord}")
#        print(f"y_coord len = {len(y_coord)}; y_coords = {y_coord}")
# 
        sc_name, sc_ver, pk_ver = get_science_code(infile)
        backscatter = np.ma.masked_invalid(data, copy=True)

        data_array = xr.Dataset({
            'y': y_coord,
            'x': x_coord, 

            # FIX ME - units should be determined by the file being processed
            'backscatter': (('y','x'), backscatter.filled(0.0),{'units':'gamma0 power','grid_mapping':cfg_p['crs_wkt'].lower()}),

            cfg_p['crs_wkt'] : cfg_p['crs_wkt'].lower(),
            'product_type': prod_type,
            'granule': granule,
            'product_name' : os.path.basename(infile),
            'start_time' : datetime.strptime(granule[17:32],"%Y%m%dT%H%M%S"),
            'end_time' : datetime.strptime(granule[33:48],"%Y%m%dT%H%M%S"),
            'platform' : granule[0:3], 
            'processing_date' : proc_dt,
            'science_code' : sc_name, 
            'science_code_version' : sc_ver,
            'package_version' : pk_ver,
            'fill_value' : saa.get_NoData_value(infile),
            'polarization' : get_pol(infile),
            'radiation_frequency' : 3.0/0.555,
            'radiation_wavelength' : 3.0/((3.0/0.555)*10),
            'sensor_band_identifier' : 'C',
            'x_spacing' : pix_x,
            'y_spacing' : pix_y,
            'scale' : otype})

        if not started:
            initialize_metadata(data_array,cfg,cfg_x,cfg_y,cfg_p,cfg_rf,cfg_rw)
            logging.info(data_array)
            xarr3d = data_array.copy(deep=True)
            started = True
            logging.info("============================================================================")
        else:
            logging.info(data_array)
            xarr3d = xr.concat([xarr3d,data_array],dim='time')
            logging.info("============================================================================")

    xarr3d = prepare_time_dimension(xarr3d,all_time,time_0_str,outfile)
    logging.info("Writing file {}".format(outfile))
    xarr3d.to_netcdf(outfile,unlimited_dims=['time'])
    logging.info("Successful Completion!")

def prepare_time_dimension(xarr3d,all_time,time_0_str,outfile):
    xarr3d = xarr3d.assign_coords(time=all_time)
    xarr3d.time.attrs['axis'] = "T" 
    xarr3d.time.attrs['units'] = "milliseconds since {}".format(time_0_str)
    xarr3d.time.attrs['calendar'] = "proleptic_gregorian"
    xarr3d.time.attrs['long_name'] = "Time"
    return xarr3d

def initialize_metadata(xarr3d,cfg,cfg_x,cfg_y,cfg_p,cfg_rf,cfg_rw):
    proj_name = cfg_p['crs_wkt']
    for key in cfg:
         xarr3d.attrs[key] = cfg[key]
    for key in cfg_x:
         xarr3d.x.attrs[key] = cfg_x[key]
    for key in cfg_y:
         xarr3d.y.attrs[key] = cfg_y[key]
    for key in cfg_p:
         xarr3d.variables[proj_name].attrs[key] = cfg_p[key]
    for key in cfg_rw:
         xarr3d.radiation_wavelength.attrs[key] = cfg_rw[key]
    for key in cfg_rf:
         xarr3d.radiation_frequency.attrs[key] = cfg_rf[key]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="make_composite.py",
             description="Create a weighted composite mosaic from a set of S-1 RTC products",
             epilog= '''Each output pixel value is calculated using a weighting that is the inverse of the area.''')

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

    make_composite(args.outfile,args.infiles,args.path,args.otype,args.requested_pol,args.resolution))
    make_composite(args.outfile,args.infiles,args.path,args.otype)



