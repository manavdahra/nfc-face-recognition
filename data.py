import os
import time
from io import BytesIO

import boto3
from PIL import Image

import constants

s3_client = boto3.client('s3')


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def create_directory(path, access_rights):
    if os.path.isdir(path):
        return
    try:
        os.makedirs(path, access_rights)
    except OSError as e:
        print(e)
        print('Creation of the directory {} failed'.format(path))
    else:
        print('Successfully created the directory {}'.format(path))


def collect_data(child_upload):
    key = child_upload['Key']
    print(splitall(key))
    child_id = splitall(key)[2]
    print('S3 Key: {0}, child_id: {1}'.format(key, child_id))

    obj = s3_client.get_object(Bucket='skoolnet-staging', Key=key)
    img = Image.open(BytesIO(obj['Body'].read()))
    create_directory(constants.CHILD_DATASET_LOCATION_FORMAT.format(child_id), 0o755)
    print('Saving img at: {0} for child_id: {1}'.format(key, child_id))
    try:
        img.save(constants.CHILD_DATASET_LOCATION_FORMAT.format(child_id) + '/{0}.jpg'.format(time.time()))
    except OSError as e:
        print(e)
        print('Could not save file for child_id: {0}'.format(child_id))


def collect_images():
    child_uploads = s3_client.list_objects_v2(Bucket='skoolnet-staging', Prefix='upload/child/100')
    for child_upload in child_uploads['Contents']:
        collect_data(child_upload)
    print('Successfully collected images and saved to {}'.format(constants.DATASET_LOCATION))
