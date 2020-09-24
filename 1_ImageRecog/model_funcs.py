# Written by Dr Daniel Buscombe, Marda Science LLC
# for "ML Mondays", a course supported by the USGS Community for Data Integration
# and the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2020, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

## TAMUCC DATA imports
from tamucc_imports import *

##UNCOMMENT BELOW TO USE NWPU DATA
# from nwpu_imports import *

#see mlmondays blog post:
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf #numerical operations on gpu
import numpy as np #numerical operations on cpu

# set a seed for reproducibility
SEED=42
np.random.seed(SEED)
tf.random.set_seed(SEED)

#for automatically determining dataset feeding processes based on available hardware
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API


from tensorflow.keras.applications import MobileNetV2 #mobilenet v2 model, used for feature extraction
from tensorflow.keras.applications import VGG16 #vgg model, used for feature extraction
from tensorflow.keras.applications import Xception #xception model, used for feature extraction


###############################################
##### MODEL FUNCTIONS
###############################################
# learning rate function
def lrfn(epoch):
    """
    "lrfn"
    This function creates a custom piecewise linear-exponential learning rate function
    for a custom learning rate scheduler. It is linear to a max, then exponentially decays
    INPUTS: current epoch number
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay
    OUTPUTS:  the function lr with all arguments passed
    """
    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        if epoch < rampup_epochs:
            lr = (max_lr - start_lr)/rampup_epochs * epoch + start_lr
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        else:
            lr = (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr
        return lr
    return lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)


#-----------------------------------
def transfer_learning_model_vgg(num_classes, input_shape, dropout_rate=0.5):
    """
    "transfer_learning_model_vgg"
    This function creates an implementation of a convolutional deep learning model for estimating
	a discrete category based on vgg, trained using transfer learning
    (initialized using pretrained imagenet weights)
    INPUTS:
        * num_classes = number of classes (output nodes on classification layer)
        * input_shape = size of input layer (i.e. image tensor)
    OPTIONAL INPUTS:
        * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
    GLOBAL INPUTS: None
    OUTPUTS: keras model instance
    """
    EXTRACTOR = VGG16(weights="imagenet", include_top=False,
                        input_shape=input_shape)

    EXTRACTOR.trainable = False
    # Construct the head of the model that will be placed on top of the
    # the base model
    class_head = EXTRACTOR.output
    class_head = tf.keras.layers.GlobalAveragePooling2D()(class_head)
    class_head = tf.keras.layers.Dense(256, activation="relu")(class_head)
    class_head = tf.keras.layers.Dropout(dropout_rate)(class_head)
    class_head = tf.keras.layers.Dense(num_classes, activation="softmax")(class_head)

    # Create the new model
    model = tf.keras.Model(inputs=EXTRACTOR.input, outputs=class_head)

    return model

#-----------------------------------
def mobilenet_model(num_classes, input_shape, dropout_rate=0.5):
    """
    "mobilenet_model"
    This function creates an implementation of a convolutional deep learning model for estimating
	a discrete category based on mobilenet, trained from scratch
    INPUTS:
        * num_classes = number of classes (output nodes on classification layer)
        * input_shape = size of input layer (i.e. image tensor)
    OPTIONAL INPUTS:
        * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
    GLOBAL INPUTS: None
    OUTPUTS: keras model instance
    """
    EXTRACTOR = MobileNetV2(weights=None, include_top=False,
                        input_shape=input_shape)

    EXTRACTOR.trainable = True
    # Construct the head of the model that will be placed on top of the
    # the base model
    class_head = EXTRACTOR.output
    class_head = tf.keras.layers.GlobalAveragePooling2D()(class_head)
    class_head = tf.keras.layers.Dense(256, activation="relu")(class_head)
    class_head = tf.keras.layers.Dropout(dropout_rate)(class_head)
    class_head = tf.keras.layers.Dense(num_classes, activation="softmax")(class_head)

    # Create the new model
    model = tf.keras.Model(inputs=EXTRACTOR.input, outputs=class_head)

    return model

#-----------------------------------
def transfer_learning_mobilenet_model(num_classes, input_shape, dropout_rate=0.5):
    """
    "transfer_learning_mobilenet_model"
    This function creates an implementation of a convolutional deep learning model for estimating
	a discrete category based on mobilenet v2, trained using transfer learning
    (initialized using pretrained imagenet weights)
    INPUTS:
        * num_classes = number of classes (output nodes on classification layer)
        * input_shape = size of input layer (i.e. image tensor)
    OPTIONAL INPUTS:
        * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
    GLOBAL INPUTS: None
    OUTPUTS: keras model instance
    """
    EXTRACTOR = MobileNetV2(weights="imagenet", include_top=False,
                        input_shape=input_shape)

    EXTRACTOR.trainable = False
    # Construct the head of the model that will be placed on top of the
    # the base model
    class_head = EXTRACTOR.output
    class_head = tf.keras.layers.GlobalAveragePooling2D()(class_head)
    class_head = tf.keras.layers.Dense(256, activation="relu")(class_head)
    class_head = tf.keras.layers.Dropout(dropout_rate)(class_head)
    class_head = tf.keras.layers.Dense(num_classes, activation="softmax")(class_head)

    # Create the new model
    model = tf.keras.Model(inputs=EXTRACTOR.input, outputs=class_head)

    return model

#-----------------------------------
def transfer_learning_xception_model(num_classes, input_shape, dropout_rate=0.25):
    """
    "transfer_learning_xception_model"
    This function creates an implementation of a convolutional deep learning model for estimating
	a discrete category based on xception, trained using transfer learning
    (initialized using pretrained imagenet weights)
    INPUTS:
        * num_classes = number of classes (output nodes on classification layer)
        * input_shape = size of input layer (i.e. image tensor)
    OPTIONAL INPUTS:
        * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
    GLOBAL INPUTS: None
    OUTPUTS: keras model instance
    """
    EXTRACTOR = Xception(weights="imagenet", include_top=False,
                        input_shape=input_shape)

    EXTRACTOR.trainable = False
    # Construct the head of the model that will be placed on top of the
    # the base model
    class_head = EXTRACTOR.output
    class_head = tf.keras.layers.GlobalAveragePooling2D()(class_head)
    class_head = tf.keras.layers.Dense(256, activation="relu")(class_head)
    class_head = tf.keras.layers.Dropout(dropout_rate)(class_head)
    class_head = tf.keras.layers.Dense(num_classes, activation="softmax")(class_head)

    # Create the new model
    model = tf.keras.Model(inputs=EXTRACTOR.input, outputs=class_head)

    return model

#-----------------------------------
def xception_model(num_classes, input_shape, dropout_rate=0.25):
    """
    "xception_model"
    This function creates an implementation of a convolutional deep learning model for estimating
	a discrete category based on xception, trained from scratch
    INPUTS:
        * num_classes = number of classes (output nodes on classification layer)
        * input_shape = size of input layer (i.e. image tensor)
    OPTIONAL INPUTS:
        * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
    GLOBAL INPUTS: None
    OUTPUTS: keras model instance
    """
    EXTRACTOR = Xception(weights=None, include_top=False,
                        input_shape=input_shape)

    EXTRACTOR.trainable = True
    # Construct the head of the model that will be placed on top of the
    # the base model
    class_head = EXTRACTOR.output
    class_head = tf.keras.layers.GlobalAveragePooling2D()(class_head)
    class_head = tf.keras.layers.Dense(256, activation="relu")(class_head)
    class_head = tf.keras.layers.Dropout(dropout_rate)(class_head)
    class_head = tf.keras.layers.Dense(num_classes, activation="softmax")(class_head)

    # Create the new model
    model = tf.keras.Model(inputs=EXTRACTOR.input, outputs=class_head)

    return model

###===================================================
def conv_block(inp, filters=32, bn=True, pool=True):
    """
    "conv_block"
    This function generates a convolutional block
    INPUTS:
        * inp = input layer
    OPTIONAL INPUTS:
        * filters = number of convolutional filters to use
        * bn=False, use batch normalization in each convolutional layer
        * pool=True, use pooling in each convolutional layer
        * shallow=True, if False, a larger model with more convolution layers is used
    GLOBAL INPUTS: None
    OUTPUTS: keras model layer object
    """
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, activation='relu',
              kernel_initializer='he_uniform')(inp)

    # _ = SeparableConv2D(filters=filters, kernel_size=3, activation='relu')(inp) #kernel_initializer='he_uniform'
    if bn:
       x = tf.keras.layers.BatchNormalization()(x)
    if pool:
       x = tf.keras.layers.MaxPool2D()(x)
    # if drop:
    #     _ = tf.keras.layers.Dropout(0.2)(_)
    return x

###===================================================
def make_cat_model(num_classes, dropout, denseunits, base_filters, bn=False, pool=True, shallow=True):
    """
    "make_cat_model"
    This function creates an implementation of a convolutional deep learning model for estimating
	a discrete category
    INPUTS:
        * num_classes = number of classes (output nodes on classification layer)
        * dropout = proportion of neurons to randomly set to zero, after the pooling layer
        * denseunits = number of neurons in the classifying layer
        * base_filters = number of convolutional filters to use in the first layer
    OPTIONAL INPUTS:
        * bn=False, use batch normalization in each convolutional layer
        * pool=True, use pooling in each convolutional layer
        * shallow=True, if False, a larger model with more convolution layers is used
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS: keras model instance
    """
    input_layer = tf.keras.layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))

    x = conv_block(input_layer, filters=base_filters, bn=bn, pool=pool)
    x = conv_block(x, filters=base_filters*2, bn=bn, pool=pool)
    x = conv_block(x, filters=base_filters*3, bn=bn, pool=pool)
    x = conv_block(x, filters=base_filters*4, bn=bn, pool=pool)

    if shallow is False:
        x = conv_block(x, filters=base_filters*5, bn=bn, pool=pool)
        x = conv_block(x, filters=base_filters*6, bn=bn, pool=pool)

    bottleneck = tf.keras.layers.GlobalMaxPool2D()(x)
    bottleneck = tf.keras.layers.Dropout(dropout)(bottleneck)

    # for class prediction
    class_head = tf.keras.layers.Dense(units=denseunits, activation='relu')(bottleneck)
    class_head = tf.keras.layers.Dense(units=num_classes, activation='softmax', name='output')(class_head)

    model = tf.keras.models.Model(inputs=input_layer, outputs=[class_head])

    model.compile(optimizer='adam',
              loss={'output': 'categorical_crossentropy'},
              metrics={'output': 'accuracy'})

    return model
