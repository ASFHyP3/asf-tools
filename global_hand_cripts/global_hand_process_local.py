#get the list of the tile file name, product the hands for these files and save at s3.
#need to run in conda aws environment.

#the url of dem files s3://asf-dem-west/v2/COP30/2021/Copernicus_DSM_COG_10_N00_00_E006_00_DEM/
#aws s3 ls copernicus-dem-30m/2021/

#s3://asf-hand-data/GLOBAL_HAND/Copernicus_DSM_COG_10_N00_00_E006_00_HAND.tif

#bucket asf-hand-data, projname=GLOBAL_HAND

#/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/2021/Copernicus_DSM_COG_10_N24_00_E088_00_DEM/Copernicus_DSM_COG_10_N24_00_E088_00_DEM.tif -b jzhu4 -p global_hand


import subprocess

import os

import argparse

try:
    import geopandas as gpd
except:
    pip install geopandas 
    import geopandas as gpd

#import scratch

parser = argparse.ArgumentParser(description='Calcualte hand for every tile')
parser.add_argument('tilelist', type=str, help='geojson file including dem tile files')
parser.add_argument('accthresh', type=int, help='acc threshold value')
parser.add_argument('s3bucket', type=str, help='s3 bucket name')
parser.add_argument('s3prefix', type=str, help='s3 prefix')

args = parser.parse_args()

accthresh = args.accthresh

bucket = args.s3bucket

prefix = args.s3prefix

flist = args.tilelist


# read the cop30-hand.geojson  file

df = gpd.read_file(flist)

tiles = df['file_path']

# rows = tiles.size

#fid = open(flist, "r")

#content_list = fid.readlines()

#print(content_list)

for item in tiles:

    #file = item.split("\n")[0]

    file = item

    command = f"python  ~/projects/asf-tools/asf_tools/hand/scratch.py {file} -t {args.accthresh} -b {args.s3bucket} -p {args.s3prefix}"

    #os.system(command)


print("completed ...")
