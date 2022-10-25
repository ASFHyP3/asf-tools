import boto3
from asf_tools.vector import get_features

batch = boto3.client('batch')

# TODO: Update revision number for job queue 
JOB_QUEUE = 'JobDefinition-09308218e67b0b7:4'
JOB_DEFINITION = 'BatchJobQueue-vSh6SePm97I5ELZr'

HAND_GEOJSON = '/vsicurl/https://asf-hand-data.s3.amazonaws.com/cop30-hand.geojson'
tile_features = get_features(HAND_GEOJSON)

for feature in tile_features:
    dem_tile_url = feature.GetFieldAsString(0)

    batch.submit_job(
        job_name=dem_tile_name,
        job_queue=JOB_QUEUE,
        jobDefinition=JOB_DEFINITION,
        parameters={
            'dem_tile': dem_tile_url,
        },
    )
