import boto3

from asf_tools.vector import get_features

batch = boto3.client('batch', region_name='us-west-2')

# TODO: Update revision number for job definition
JOB_QUEUE = 'BatchJobQueue-vSh6SePm97I5ELZr'
JOB_DEFINITION = 'JobDefinition-09308218e67b0b7:7'

HAND_GEOJSON = 'https://asf-dem-west.s3.us-west-2.amazonaws.com/v2/cop30-2021.geojson'
tile_features = get_features(HAND_GEOJSON)

for feature in tile_features:
    dem_tile_url = feature.GetFieldAsString(0)

    batch.submit_job(
        jobName=dem_tile_url.split('/')[-2][:-4],
        jobQueue=JOB_QUEUE,
        jobDefinition=JOB_DEFINITION,
        parameters={
            'dem_tile_url': dem_tile_url,
        },
    )
