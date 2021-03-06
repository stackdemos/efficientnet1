"""Generates predictions using a stored model.

Uses trained model files to generate a prediction.
"""

from __future__ import print_function

import os, numbers

import numpy as np
import dill as dpickle
from keras.models import load_model
from seq2seq_utils import Seq2Seq_Inference


class IssueSummarizationModel(object):

    def __init__(self):
        body_pp_file = os.getenv('BODY_PP_FILE', '/data/body_preprocessor.dpkl')
        print('body_pp file {0}'.format(body_pp_file))
        with open(body_pp_file, 'rb') as body_file:
            body_pp = dpickle.load(body_file)

        title_pp_file = os.getenv('TITLE_PP_FILE', '/data/output_model.h5')
        print('title_pp file {0}'.format(title_pp_file))
        with open(title_pp_file, 'rb') as title_file:
            title_pp = dpickle.load(title_file)

        model_file = os.getenv('MODEL_FILE', 'seq2seq_model_tutorial.h5')
        print('model file {0}'.format(model_file))
        self.model = Seq2Seq_Inference(encoder_preprocessor=body_pp,
                                    decoder_preprocessor=title_pp,
                                    seq2seq_model=load_model(model_file))

    def predict(self, input_text, feature_names):  # pylint: disable=unused-argument
        return np.asarray([[self.model.generate_issue_title(body[0])[1]] for body in input_text])
    
        get_custom_objects().update({
            'ConvKernalInitializer': ConvKernalInitializer,
            'Swish': Swish,
            'DropConnect':DropConnect
        })
        if not self.model:
            self.model = load_model(self.model_file)
        # Do any preprocessing
        prediction = self.model.predict(data=X)
        # Do any postprocessing
        return prediction


