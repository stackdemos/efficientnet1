#!/usr/bin/env python

"""
Simple app that parses predictions from a trained model and displays them.
"""

from flask_json import json_response
from uptime import uptime
from random import sample
from os import environ
import ptvsd

import os
import re
import random

import requests
import pandas as pd
from flask import Flask, json, render_template, request, g, jsonify
from requests.auth import HTTPBasicAuth

import kfp.dsl as dsl
import kfp.gcp as gcp
import pandas as pd
from kfp.compiler import Compiler
from kfp.components import load_component_from_file
from os import environ
from datetime import datetime
import boto3, kfp

SAMPLE_DATA = os.environ.get('SAMPLE_DATA', '/data/sample.csv')

OAUTH_KEY = os.environ.get('OAUTH_KEY', '')
OAUTH_SECRET = os.environ.get('OAUTH_SECRET', '')
SERVER_ADDR = os.environ.get('SERVER_ADDR', 'seldon-seldon-apiserver.kubeflow:8080')
APPLICATION_NAME= 'efficientnet1'

application = Flask(__name__)
application.config.from_pyfile(f"conf/{application.config['ENV']}.py")

# Remote debug settings
# We can debug debug or code reload: cannot do both
# Reload controlled as environment variable
if application.debug and int(environ.get("FLASK_RUN_RELOAD", 0)) == 0:
    application.logger.info("Starting `ptvs` on port: 3000")
    ptvsd.enable_attach(address=('0.0.0.0', 3000))


def get_issue_body(issue_url):
    issue_url = re.sub('.*github.com/', 'https://api.github.com/repos/', issue_url)
    return requests.get(issue_url, headers={'Authorization': 'token {}'.format('GITHUB_TOKEN')}).json()['body']

def get_token():
    """returns bearer token for seldon deployment
    """
    response = requests.post(
                f"http://{SERVER_ADDR}/oauth/token",
                auth=HTTPBasicAuth(OAUTH_KEY, OAUTH_SECRET),
                data={'grant_type': 'client_credentials'})
    return response.json()["access_token"]



@application.route('/')
def index():
    """
    return the rendered index.html template
    """
    return render_template("index.html")



@application.route("/summary", methods=['POST'])
def summary():
    """Main form.
    Test object detection
    Send image URL to the model to perform object detection
    """
    if request.method == 'POST':
        test_image = request.form["test_image"]
        print(f'Serve model : {test_image}')
        url = f"http://{SERVER_ADDR}/api/v0.1/predictions"
        token = get_token()
        headers = {
            'content-type': 'application/json',
            'Authorization': f"Bearer {token}"
        }
        json_data = {"data": {"ndarray": [[test_image]]}}
        response = requests.post(url=url, headers=headers, data=json.dumps(json_data))
        response_json = json.loads(response.text)
        issue_summary = response_json["data"]["ndarray"][0][0]
        return jsonify({'summary': issue_summary, 'body': test_image})

    return ('', 204)


@application.route("/get_experiment", methods=['POST'])
def get_experiment():
    dataset_dir = request.form["dataset_dir"]
    epochs = request.form["epochs"]
    print(f'Get experiment : {dataset_dir} {epochs}')
    client = kfp.Client()
    try:
        exp = client.get_experiment(experiment_name=APPLICATION_NAME)
    except:
        exp = client.create_experiment(APPLICATION_NAME)
    print (f"Experiment name: {exp.name}")
    return jsonify({'id': exp.id, 'experiment_name': exp.name})

@application.route("/train", methods=['POST'])
def train():
    DATASET_DIR = request.form['dataset_dir']
    EPOCHS = request.form['epochs']
    DATASET_NAME = 'normal_pneumothorax'
    LABELS = 'normal,pneumothorax'
    REMOTE_MINIO_SERVER = '206.189.86.150:32782'
    ACCESS_KEY = 'admin'
    SECRET_KEY = 'deeplearning'
    MODEL_VERSION = '1'
    MODEL_DIR = '/mnt/s3/models/'
    MODEL_FNAME = 'pneumothorax_03_13_50.h5'
    
    
    print(f'Training model : {DATASET_DIR} {EPOCHS}')
    client = kfp.Client()
    try:
        exp = client.get_experiment(experiment_name=APPLICATION_NAME)
    except:
        exp = client.create_experiment(APPLICATION_NAME)
    # Use the following API to find Pipeline Id for your Kubeflow pipeline: https://kubeflow.svc.ml1.demo51.superhub.io/pipeline/apis/v1beta1/pipelines
    print (f"Experiment name: {exp.name}")    
    run = client.run_pipeline(exp.id, f'Training model : {datetime.now():%m%d-%H%M}',
                          params={
                              'datasetDir': DATASET_DIR,
                              'datasetName': DATASET_NAME,
                              'labels': LABELS,
                              'remoteMinioServer': REMOTE_MINIO_SERVER,
                              'accessKey': ACCESS_KEY,
                              'secretKey': SECRET_KEY,
                              'batchSize': 32,
                              'width': 150,
                              'height': 150,
                              'epochs': 5,
                              'dropoutRate': 0.2,
                              'learningRate': 0.00002,
                              'trainInput': os.path.join(DATASET_DIR, DATASET_NAME),
                              'modelVersion': MODEL_VERSION,
                              'modelDir': MODEL_DIR,
                              'modelFname': MODEL_FNAME,
                          }, pipeline_id='75fdc7c1-1b49-4a8c-ac96-13b5af4a1364')
    return (f'Submitted {run.name} : {run.id}')

@application.route('/status')
def status():
    """
    returns healthcheck and uptime
    """
    return json_response(
        200,
        status="ok",
        uptime=uptime(),
    )


if __name__ == "__main__":
    application.run(
      host=application.config.get('HOST'),
      port=application.config.get('PORT')
    )
