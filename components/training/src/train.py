import argparse, os, glob
import numpy as np
import logging
# reduce tensorflow warnings in logs
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tempfile
import glob
import shutil
import pandas as pd
from datetime import datetime
import itertools
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard

# Options: EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3
# Higher the number, the more complex the model is.
from efficientnet import EfficientNetB0 as Net
from efficientnet import center_crop_and_resize, preprocess_input
from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer

from utils import (get_value, get_value_as_int, get_value_as_float, copy)

def read_input(original_dataset_dir, dataset_name, labels, test_size=0.2):
    """Read input data and split it into train and test."""
        
    print(f"1. Original_dataset_dir:{original_dataset_dir} dataset_name:{dataset_name} labels: {labels}")
    cat_images_list = []
    labels_arr = labels.split(',')
    labels_list = []
    FILETYPES = ('*.jpg', '*.jpeg', '*.png')

    for cat in labels_arr:
        cat_images = [glob.glob(os.path.join(original_dataset_dir,cat,e)) for e in FILETYPES]  
        cat_images = list(itertools.chain(*cat_images))
        print("total " + cat + " images: {}".format(len(cat_images)))
        labels_list += [cat]*len(cat_images)
        cat_images_list.append(cat_images)

    all_images_list = list(itertools.chain(*cat_images_list))
    
    assert len(all_images_list) == len(labels_list)

    X_train, X_test, y_train, y_test = train_test_split(
        all_images_list, labels_list, test_size=test_size, random_state=1)

    num_train = len(X_train)
    num_test = len(X_test)

    train_df = pd.DataFrame({
        'filename': X_train,
        'label': y_train
    })

    val_df = pd.DataFrame({
        'filename': X_test,
        'label': y_test
    })

    return train_df, val_df


# Parsing flags.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size")
parser.add_argument("--width")
parser.add_argument("--height")
parser.add_argument("--epochs")
parser.add_argument("--dropout_rate")
parser.add_argument("--learning_rate")
parser.add_argument("--dataset_name")
parser.add_argument("--train_input")
parser.add_argument("--model_version")
parser.add_argument("--model_dir")
parser.add_argument("--model_fname")
parser.add_argument("--labels")
parser.add_argument("--tempfile", default=True)
args = parser.parse_args()
print(args)

LOG_DIR = args.model_dir + '/logs'
tensorboard_callback = TensorBoard(log_dir=LOG_DIR)

original_dataset_dir = args.train_input
height = int(args.height)
width = int(args.width)
batch_size = int(args.batch_size)
dropout_rate = float(args.dropout_rate)
epochs = int(args.epochs)
model_dir = args.model_dir
model_file = model_dir + args.model_version + "/" + args.model_fname
learning_rate = float(args.learning_rate)
                    
train_df, val_df = read_input(args.train_input, args.dataset_name, args.labels)

                                       
#####################################
# Train the model using EfficientNet.
#####################################
input_shape = (height, width, 3)

num_train = len(train_df.index)
num_test = len(val_df.index)

# loading pretrained conv base model
conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=original_dataset_dir,
        x_col="filename",
        y_col="label",
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_dataframe(
            dataframe=val_df,
            directory=original_dataset_dir,
            x_col="filename",
            y_col="label",
            target_size=(height, width),
            batch_size=batch_size,
            class_mode='categorical')

model = models.Sequential()
model.add(conv_base)
model.add(layers.GlobalMaxPooling2D(name="gap"))
# model.add(layers.Flatten(name="flatten"))
if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))
# model.add(layers.Dense(256, activation='relu', name="fc1"))
model.add(layers.Dense(2, activation='softmax', name="fc_out"))

conv_base.trainable = False

model.compile(loss='categorical_crossentropy',
            optimizer=optimizers.RMSprop(lr=learning_rate),
            metrics=['acc'])
print('Training model...')
history = model.fit_generator(
    train_generator,
    steps_per_epoch= num_train //batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps= num_test //batch_size,
    verbose=1)        #        callbacks=[tensorboard_callback],
#         use_multiprocessing=True,
#         workers=4)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_x = range(len(acc))

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'multiply_16':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='categorical_crossentropy',
            optimizer=optimizers.RMSprop(lr=2e-5),
            metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch= num_train //batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps= num_test //batch_size,
    verbose=1)

#############
# Save model.
#############
#    print('Saving model to remote...')
#    mc_remote = Minio(REMOTE_MINIO_SERVER,
#                  access_key=ACCESS_KEY,
#                  secret_key=SECRET_KEY,
#                  secure=False)
#
#    logging.info("Model export success: %s", model_fname)
#    with tempfile.NamedTemporaryFile(suffix='.h5') as fp:
#        model.save(fp.name)
#        try:
#            print(mc_remote.fput_object('models', os.path.join(model_version, model_fname),fp.name))
#        except ResponseError as err:
#            print(err)

print("Saving model...")

if args.tempfile:
    print('Saving model to local...')
    # Workaround because of: at present goofys support only parallel write
    # see: https://github.com/kahing/goofys/issues/298
    # TODO configure h5py to write sequentially
    # TODO consider other flex driver
    _, fname = tempfile.mkstemp('.h5')
    print(f"Saving to {fname}")
    model.save(fname)
    copy(fname, model_file)
    to_dir = dirname = os.path.dirname(model_file)
    print("Saving weights...")
    for f in glob.iglob('/tmp/*.hdf5'):
        copy(f, to_dir)
    print('Saved models')
else:
    print(f"Saving to {model_file}")
    seq2seq_Model.save(model_file)
print("Done!")
