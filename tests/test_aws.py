import pytest
from botocore.stub import ANY, Stubber

from asf_tools import aws


@pytest.fixture(autouse=True)
def s3_stubber():
    with Stubber(aws.S3_CLIENT) as stubber:
        yield stubber
        stubber.assert_no_pending_responses()


def test_get_tag_set():
    assert aws.get_tag_set() == {
        'TagSet': [
            {
                'Key': 'file_type',
                'Value': 'product'
            }
        ]
    }


def test_get_content_type():
    assert aws.get_content_type('foo') == 'application/octet-stream'
    assert aws.get_content_type('foo.asfd') == 'application/octet-stream'
    assert aws.get_content_type('foo.txt') == 'text/plain'
    assert aws.get_content_type('foo.zip') == 'application/zip'
    assert aws.get_content_type('foo/bar.png') == 'image/png'


def test_upload_file_to_s3(tmp_path, s3_stubber):
    expected_params = {
        'Body': ANY,
        'Bucket': 'myBucket',
        'Key': 'myFile.zip',
        'ContentType': 'application/zip',
    }
    tag_params = {
        'Bucket': 'myBucket',
        'Key': 'myFile.zip',
        'Tagging': {
            'TagSet': [
                {'Key': 'file_type', 'Value': 'product'}
            ]
        }
    }
    s3_stubber.add_response(method='put_object', expected_params=expected_params, service_response={})
    s3_stubber.add_response(method='put_object_tagging', expected_params=tag_params, service_response={})

    file_to_upload = tmp_path / 'myFile.zip'
    file_to_upload.touch()
    aws.upload_file_to_s3(file_to_upload, 'myBucket')


def test_upload_file_to_s3_with_prefix(tmp_path, s3_stubber):
    expected_params = {
        'Body': ANY,
        'Bucket': 'myBucket',
        'Key': 'myPrefix/myFile.txt',
        'ContentType': 'text/plain',
    }
    tag_params = {
        'Bucket': 'myBucket',
        'Key': 'myPrefix/myFile.txt',
        'Tagging': {
            'TagSet': [
                {'Key': 'file_type', 'Value': 'product'}
            ]
        }
    }
    s3_stubber.add_response(method='put_object', expected_params=expected_params, service_response={})
    s3_stubber.add_response(method='put_object_tagging', expected_params=tag_params, service_response={})
    file_to_upload = tmp_path / 'myFile.txt'
    file_to_upload.touch()
    aws.upload_file_to_s3(file_to_upload, 'myBucket', 'myPrefix')


def test_get_path_to_s3_file(s3_stubber):
    expected_params = {
        'Bucket': 'myBucket',
        'Prefix': 'myPrefix',
    }
    service_response = {
        'Contents': [
            {'Key': 'myPrefix/foo.txt'},
            {'Key': 'myPrefix/foo.nc'},
            {'Key': 'myPrefix/foo.txt'},
            {'Key': 'myPrefix/bar.nc'},
        ],
    }

    s3_stubber.add_response(method='list_objects_v2', expected_params=expected_params,
                            service_response=service_response)
    assert aws.get_path_to_s3_file('myBucket', 'myPrefix', file_type='.nc') == '/vsis3/myBucket/myPrefix/foo.nc'

    s3_stubber.add_response(method='list_objects_v2', expected_params=expected_params,
                            service_response=service_response)
    assert aws.get_path_to_s3_file('myBucket', 'myPrefix', file_type='.txt') == '/vsis3/myBucket/myPrefix/foo.txt'

    s3_stubber.add_response(method='list_objects_v2', expected_params=expected_params,
                            service_response=service_response)
    assert aws.get_path_to_s3_file('myBucket', 'myPrefix', file_type='.csv') is None
