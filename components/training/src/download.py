#!/usr/bin/env python

import argparse
import numpy as np
import logging
import os
import tempfile
import pandas as pd
from datetime import datetime
import itertools

import logging
from minio import Minio
from minio.error import ResponseError

# Parsing args.
parser = argparse.ArgumentParser(description="Download training and test data sets.")
parser.add_argument("--dataset_dir")
parser.add_argument("--dataset_name")
parser.add_argument("--labels")
parser.add_argument("--remote_minio_server")
parser.add_argument("--access_key")
parser.add_argument("--secret_key")
args = parser.parse_args()
print(args)

DATASET_DIR = args.dataset_dir
DATASET_NAME = args.dataset_name
LABELS = args.labels
REMOTE_MINIO_SERVER = args.remote_minio_server
ACCESS_KEY = args.access_key
SECRET_KEY = args.secret_key

# create local directories
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.join(DATASET_DIR, DATASET_NAME), exist_ok=True)    

for x in LABELS.split(','):
    os.makedirs(os.path.join(DATASET_DIR, DATASET_NAME, x), exist_ok=True)

if not os.path.exists(DATASET_DIR):
    print(f"Connect to remote minio server for downloading files: {REMOTE_MINIO_SERVER}")
    # connect to remote minio server for downloading files
    mc_remote = Minio(REMOTE_MINIO_SERVER,
                  access_key=ACCESS_KEY,
                  secret_key=SECRET_KEY,
                  secure=False)

    objects = mc_remote.list_objects('datasets', prefix=DATASET_NAME,
                                  recursive=True)
    for obj in objects:
        try:
            data = mc_remote.get_object('datasets', obj.object_name)
            with open(os.path.join(dataset_dir, obj.object_name), 'wb') as file_data:
                for d in data.stream(32*1024):
                    file_data.write(d)
        except ResponseError as err:
            print(err)
else:
    print(f"Exiting, dataset exists: {DATASET_DIR}")
    