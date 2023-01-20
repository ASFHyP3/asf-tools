import logging
from pathlib import Path
from typing import Union

import boto3

S3_CLIENT = boto3.client('s3')
log = logging.getLogger(__name__)


def get_tag_set() -> dict:
    tag_set = {
        'TagSet': [
            {
                'Key': 'file_type',
                'Value': 'product'
            }
        ]
    }
    return tag_set


def upload_file_to_s3(path_to_file: Union[str, Path], bucket: str, prefix: str = ''):
    path_to_file = Path(path_to_file)
    if not path_to_file.suffix == '.geojson':
        raise ValueError(f'Expected geojson file. Got {path_to_file.suffix}. Exiting.')

    key = str(Path(prefix) / path_to_file)
    extra_args = {'ContentType': 'application/geo+json'}

    log.info(f'Uploading s3://{bucket}/{key}')
    S3_CLIENT.upload_file(str(path_to_file), bucket, key, extra_args)

    tag_set = get_tag_set()

    S3_CLIENT.put_object_tagging(Bucket=bucket, Key=key, Tagging=tag_set)


def get_path_to_s3_file(bucket_name, bucket_prefix, file_type: str):
    result = S3_CLIENT.list_objects_v2(Bucket=bucket_name, Prefix=bucket_prefix)
    for s3_object in result['Contents']:
        key = s3_object['Key']
        if key.endswith(file_type):
            return f'/vsicurl/https://{bucket_name}.s3.amazonaws.com/{key}'
