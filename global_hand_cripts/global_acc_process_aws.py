#get the list of the tile file name, product the hands for these files and save at s3.
#need to run in conda aws environment.

import subprocess

import argparse

parser = argparse.ArgumentParser(description='Calcualte acc for every tile')

parser.add_argument('projname', type=str, help='project name')

parser.add_argument('tilelist', type=str, help='list of dem tile files')


args = parser.parse_args()

projname = args.projname

with open(args.tilelist,'r') as f:

    tlist =  f.readlines()
    
    job_no = 0
   
    for tile in tlist:
        job_no =job_no + 1

        job_no_str = str(job_no).zfill(5)

        tilename = tile.rstrip('\n')
    
        file=f"{tilename}.tif"
        
        job_name = f"{projname}_{tilename}_{job_no_str}"

        command = f"aws batch submit-job --job-name {job_name} --job-queue jzhu-batch-job-queue --job-definition jzhu-asf-tools-acc-job-definition:1 --container-overrides command=/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/2021/{tilename}/{file},10,{projname}"
        
        command2 ="/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/2021/Copernicus_DSM_COG_10_N24_00_E088_00_DEM/Copernicus_DSM_COG_10_N24_00_E088_00_DEM.tif"
        print(command)
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)


print("completed ...")
