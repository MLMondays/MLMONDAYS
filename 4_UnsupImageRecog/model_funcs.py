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

## TAMUCC
from tamucc_imports import *

##UNCOMMENT BELOW TO USE NWPU DATA
# from nwpu_imports import *

#see mlmondays blog post:
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

##calcs
import tensorflow as tf #numerical operations on gpu
import numpy as np #numerical operations on cpu

from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import class_weight #utility for computinh normalised class weights


SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

import tensorflow.keras.backend as K

###############################################################
### MODEL FUNCTIONS
###############################################################

class EmbeddingModel(tf.keras.Model):
    """
    # code modified from https://keras.io/examples/vision/metric_learning/
    "EmbeddingModel"
    This class allows an embedding model (an get_embedding_model or get_large_embedding_model instance)
    to be trainable using the conventional model.fit(), whereby it can be passed another class
    that provides batches of data examples in the form of anchors, positives, and negatives
    INPUTS: None
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: model training metrics
    """
    def train_step(self, data):
        # Note: Workaround for open issue, to be removed.
        if isinstance(data, tuple):
            data = data[0]
        anchors, positives = data[0], data[1]

        with tf.GradientTape() as tape:
            # Run both anchors and positives through model.
            anchor_embeddings = self(anchors, training=True)
            positive_embeddings = self(positives, training=True)

            # Calculate cosine similarity between anchors and positives. As they have
            # been normalised this is just the pair wise dot products.
            similarities = tf.einsum(
                "ae,pe->ap", anchor_embeddings, positive_embeddings
            )

            # Since we intend to use these as logits we scale them by a temperature.
            # This value would normally be chosen as a hyper parameter.
            temperature = 0.3 # 0.2
            similarities /= temperature

            # We use these similarities as logits for a softmax. The labels for
            # this call are just the sequence [0, 1, 2, ..., num_classes] since we
            # want the main diagonal values, which correspond to the anchor/positive
            # pairs, to be high. This loss will move embeddings for the
            # anchor/positive pairs together and move all other pairs apart.
            sparse_labels = tf.range(num_classes)
            # sparse_labels[0] = sparse_labels[0]*class_weights[0]
            # sparse_labels[1] = sparse_labels[1]*class_weights[1]
            loss = self.compiled_loss(sparse_labels, similarities)

        # Calculate gradients and apply via optimizer.
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (specifically the one for the loss value).
        self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}

#---------------------------------------------------
def get_large_embedding_model(TARGET_SIZE, num_classes, num_embed_dim):
    """
    # code modified from https://keras.io/examples/vision/metric_learning/
    "get_large_embedding_model"
    This function makes an instance of a larger embedding model, which is a keras sequential model
    consisting of 5 convolutiional blocks, average 2d pooling, and an embedding layer
    INPUTS:
        * model [keras model]
        * X_train [list]
        * ytrain [list]
        * num_dim_use [int]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * knn [sklearn knn model]
    """
    inputs = tf.keras.layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, activation="relu")(inputs) #
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu")(inputs) #
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu")(x) #32
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu")(x) #64
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, activation="relu")(x) #64
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    embeddings = tf.keras.layers.Dense(units = num_embed_dim, activation=None)(x)
    #embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    model = EmbeddingModel(inputs, embeddings)
    return model

#---------------------------------------------------
def get_embedding_model(TARGET_SIZE, num_classes, num_embed_dim):
    """
    # code modified from https://keras.io/examples/vision/metric_learning/
    "get_embedding_model"
    This function makes an instance of an embedding model, which is a keras sequential model
    consisting of 3 convolutiional blocks, average 2d pooling, and an embedding layer
    INPUTS:
        * model [keras model]
        * X_train [list]
        * ytrain [list]
        * num_dim_use [int]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * knn [sklearn knn model]
    """
    inputs = tf.keras.layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu")(inputs) #
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu")(x) #32
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu")(x) #64
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    embeddings = tf.keras.layers.Dense(units = num_embed_dim, activation=None)(x)

    # according to matt kelcey, normalizing embeddings during training is not optimal
    # even though this does help for embeddings on test samples. Not sure if that is universally true
    # so left this commented - treat as a hyperparameter!
    #embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    model = EmbeddingModel(inputs, embeddings)
    return model

#---------------------------------------------------
def fit_knn_to_embeddings(model, X_train, ytrain, n_neighbors):
    """
    "fit_knn_to_embeddings"
    This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
    using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
    saving out to the provided filename, cm_filename
    INPUTS:
        * model [keras model]
        * X_train [list]
        * ytrain [list]
        * num_dim_use [int]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * knn [sklearn knn model]
    """
    embeddings = model.predict(X_train)
    del X_train
    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(embeddings.numpy(), ytrain)
    return knn



###############################################################
### TRAINING FUNCTIONS
###############################################################

def weighted_binary_crossentropy(zero_weight, one_weight):
    """
    "weighted_binary_crossentropy"
    This function computes weighted binary crossentropy loss
    INPUTS:
        * zero_weight [float]: weight for the zero class
        * one_weight [float]: weight for the one class
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:  the function wbce with all arguments passed
    """
    def wbce(y_true, y_pred):

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred, from_logits=True)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return wbce

#---------------------------------------------------
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
