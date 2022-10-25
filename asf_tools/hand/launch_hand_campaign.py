import boto3

batch = boto3.client('batch')

# TODO fill in after batch environment is set up
JOB_QUEUE = ''
JOB_DEFINITION = ''

# TODO populate tile list from s3://...
tile_list = []

for dem_tile_name in tile_list:
    dem_tile_url = f'/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/2021/{dem_tile_name}/{dem_tile_name}.tif"'

    batch.submit_job(
        job_name=dem_tile_name,
        job_queue=JOB_QUEUE,
        jobDefinition=JOB_DEFINITION,
        parameters={
            'dem_tile': dem_tile_url,
        },
    )
