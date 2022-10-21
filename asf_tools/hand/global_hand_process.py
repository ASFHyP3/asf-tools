#get the list of the tile file name, product the hands for these files and save at s3.
#need to run in conda aws environment.

#the url of dem files s3://asf-dem-west/v2/COP30/2021/Copernicus_DSM_COG_10_N00_00_E006_00_DEM/
#aws s3 ls copernicus-dem-30m/2021/

#s3://asf-hand-data/GLOBAL_HAND/Copernicus_DSM_COG_10_N00_00_E006_00_HAND.tif

#bucket asf-hand-data, projname=GLOBAL_HAND

#/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/2021/Copernicus_DSM_COG_10_N24_00_E088_00_DEM/Copernicus_DSM_COG_10_N24_00_E088_00_DEM.tif -b jzhu4 -p global_hand


import subprocess

import argparse

import geopandas as gpd

parser = argparse.ArgumentParser(description='Calcualte hand for every tile')
parser.add_argument('geojson', type=str, help='geojson file including dem tile files')
parser.add_argument('s3bucket', type=str, help='s3 bucket name')
parser.add_argument('s3prefix', type=str, help='s3 prefix')

args = parser.parse_args()

bucket = args.s3bucket

prefix = args.s3prefix

geojsonfile = args.geojson

df = gpd.read_file(geojsonfile)

tiles = df['file_path']

rows = tiles.size

for i in range(rows):
    
    if i > 3:

        break

    job_no_str = str(i).zfill(5)

    fpath = tiles[i]

    fname = fpath.split("/")[-1]

    tilename = fname.split(".tif")[0]

    job_name = f"{prefix}_{tilename}_{job_no_str}"

    # for produce_hand_jzhu.py

    command = f"aws batch submit-job \
            --job-name {job_name} \
            --job-queue jzhu-batch-job-queue \
            --job-definition jzhu-asf-tools-job-definition:3 \
            --container-overrides command={fpath} {bucket} {prefix}"

    # for produce_hand_jhk.py
    #command = f"aws batch submit-job \
    #--job-name {job_name} \
    #--job-queue jzhu-batch-job-queue \
    #--job-definition jzhu-asf-tools-job-definition:4 \
    #--container-overrides command='{fpath} -b {bucket} -p {prefix}'"

    print(command)

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)


print("completed ...")
