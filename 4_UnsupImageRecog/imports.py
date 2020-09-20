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
from sklearn.manifold import TSNE #for data dimensionality reduction / viz.
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

##plots
import matplotlib.pyplot as plt #for plotting
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix #compute confusion matrix from vectors of observed and estimated labels
import seaborn as sns #extended functionality / style to matplotlib plots
from matplotlib.offsetbox import OffsetImage, AnnotationBbox #for visualizing image thumbnails plotted as markers

##utils
from collections import defaultdict
from PIL import Image
from collections import Counter
#keras functions for early stopping and model weights saving
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight #utility for computinh normalised class weights
from tensorflow.keras import backend as K #access to keras backend functions
from collections import defaultdict

##i/o
import pandas as pd #for data wrangling. We just use it to read csv files
import json, shutil #json for class file reading, shutil for file copying/moving


SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

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
def fit_knn_to_embeddings(model, X_train, ytrain, num_dim_use, n_neighbors):
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
    knn.fit(embeddings.numpy()[:,:num_dim_use], ytrain)
    return knn


###############################################################
### PLOT FUNCTIONS
###############################################################

#-----------------------------------
def conf_mat_filesamples(model, knn, sample_filenames, num_classes, num_dim_use, CLASSES):
    """
    "conf_mat_filesamples"
    This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
    using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
    saving out to the provided filename, cm_filename
    INPUTS:
        * model [keras model]
        * knn [sklearn knn model]
        * sample_filenames [list] of strings
        * num_classes [int]
        * num_dim_use [int]
        * CLASSES [list] of strings: class names
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * cm [ndarray]: confusion matrix
    """
    ## compute confusion matric for all samples
    cm = np.zeros((num_classes, num_classes))

    y_obs = []
    y_est = []
    for f in sample_filenames:
        # read image and convert to 32-bit tensor
        image = tf.cast(file2tensor(f), np.float32)
        # get the embeddings from the neural network model
        embeddings_sample = model.predict(tf.expand_dims(image, 0))
        # get class numeric code prediction from the k-nearest neighbours model
        est_class_idx = knn.predict(embeddings_sample[:,:num_dim_use])[0]
        y_est.append(est_class_idx)
        obs_class = f.split('/')[-1].split('_IMG')[0] #could this be a lambda function passed to the function as an argument?
        # get numeric code from class name
        class_idx = [i for i,c in enumerate(CLASSES) if c.decode()==obs_class][0]
        y_obs.append(class_idx)
        cm[class_idx, est_class_idx] += 1

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


#-----------------------------------
def p_confmat(labs, preds, cm_filename, CLASSES, thres = 0.1):
    """
    "p_confmat"
    This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
    using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
    saving out to the provided filename, cm_filename
    INPUTS:
        * labs [ndarray]: 1d vector of labels
        * preds [ndarray]: 1d vector of model predicted labels
        * cm_filename [string]: filename to write the figure to
        * CLASSES [list] of strings: class names
    OPTIONAL INPUTS:
        * thres [float]: threshold controlling what values are displayed
    GLOBAL INPUTS: None
    OUTPUTS: None (figure printed to file)
    """
    cm = confusion_matrix(labs, preds)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cm[cm<thres] = 0

    plt.figure(figsize=(15,15))
    sns.heatmap(cm,
        annot=True,
        cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True))

    tick_marks = np.arange(len(CLASSES))+.5
    plt.xticks(tick_marks, [c.decode() for c in CLASSES], rotation=90,fontsize=12)
    plt.yticks(tick_marks, [c.decode() for c in CLASSES],rotation=0, fontsize=12)

    # plt.show()
    plt.savefig(cm_filename,
                dpi=200, bbox_inches='tight')
    plt.close('all')

    print("Average true positive rate across %i classes: %.3f" % (len(CLASSES), np.mean(np.diag(cm))))


###############################################################
### DATA FUNCTIONS
###############################################################
#-----------------------------------
def read_classes_from_json(json_file):
    """
    "read_classes_from_json"
    This function reads the contents of a json file enumerating classes
    INPUTS:
        * json_file [string]: full path to the json file
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * CLASSES [list]: list of classesd as byte strings
    """
    with open(json_file) as f:
        class_dict = json.load(f)

    # string names
    CLASSES = [class_dict[k] for k in class_dict.keys()]
    #bytestrings names
    CLASSES = [c.encode() for c in CLASSES]
    return CLASSES


## test using imgae read from file
#-----------------------------------
def file2tensor(f):
    """
    "file2tensor"
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained mobilenet or vgg model
    (the imagery is standardized depedning on target model framework)
    INPUTS:
        * f [string] file name of jpeg
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
        * im [tensor array]: standardized image
    GLOBAL INPUTS: TARGET_SIZE
    """
    bits = tf.io.read_file(f)
    image = tf.image.decode_jpeg(bits)

    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = TARGET_SIZE
    th = TARGET_SIZE
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                 )

    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    image = tf.cast(image, tf.uint8) #/ 255.0

    return image

###############################################################
### TFRECORD FUNCTIONS
###############################################################

#-----------------------------------
def resize_and_crop_image(image, label):
    """
    "resize_and_crop_image"
    This function crops to square and resizes an image
    The label passes through unmodified
    INPUTS:
        * image [tensor array]
        * label [int]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * label [int]
    """
    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = TARGET_SIZE
    th = TARGET_SIZE
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                 )
    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    return image, label

#-----------------------------------
def recompress_image(image, label):
    """
    "recompress_image"
    This function takes an image encoded as a byte string
    and recodes as an 8-bit jpeg
    Label passes through unmodified
    INPUTS:
        * image [tensor array]
        * label [int]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * image [tensor array]
        * label [int]
    """
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
    return image, label

#-----------------------------------
"""
These functions cast inputs into tf dataset 'feature' classes
There is one for bytestrings (images), one for floats (not used here) and one for ints (labels)
"""
def _bytestring_feature(list_of_bytestrings):
    """
    "_bytestring_feature"
    cast inputs into tf dataset 'feature' classes
    INPUTS:
        * list_of_bytestrings
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS: tf.train.Feature example
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints):
    """
    "_int_feature"
    cast inputs into tf dataset 'feature' classes
    INPUTS:
        * list_of_ints
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: tf.train.Feature example
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats):
    """
    "_float_feature"
    cast inputs into tf dataset 'feature' classes
    INPUTS:
        * list_of_floats
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: tf.train.Feature example
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

#-----------------------------------
def to_tfrecord(img_bytes, label, CLASSES):
    """
    "to_tfrecord"
    This function creates a TFRecord example from an image byte string and a label feature
    INPUTS:
        * img_bytes
        * label
        * CLASSES
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: tf.train.Feature example
    """
    class_num = np.argmax(np.array(CLASSES)==label)
    feature = {
      "image": _bytestring_feature([img_bytes]), # one image in the list
      "class": _int_feature([class_num]),        # one class in the list
              }
    return tf.train.Example(features=tf.train.Features(feature=feature))

#-----------------------------------
def read_tfrecord(example):
    """
    "read_tfrecord"
    This function reads an example from a TFrecord file into a single image and label
    INPUTS:
        * TFRecord example object
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * class_label [tensor int]
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32)/ 255.0
    image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])

    class_label = tf.cast(example['class'], tf.int32)

    return image, class_label

#-----------------------------------
def read_image_and_label(img_path):
    """
    "read_image_and_label"
    This function reads a jpeg image from a provided filepath
    and extracts the label from the filename (assuming the class name is
    before "_IMG" in the filename)
    INPUTS:
        * img_path [string]: filepath to a jpeg image
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * image [tensor array]
        * class_label [tensor int]
    """
    bits = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(bits)

    label = tf.strings.split(img_path, sep='/')
    label = tf.strings.split(label[-1], sep='_IMG')

    return image,label[0]

#-----------------------------------
def get_dataset_for_tfrecords(recoded_dir, shared_size):
    """
    "get_dataset_for_tfrecords"
    This function reads a list of TFREcord shard files,
    decode the images and label
    resize and crop the image to TARGET_SIZE
    and create batches
    INPUTS:
        * recoded_dir
        * shared_size
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * tf.data.Dataset object
    """
    tamucc_dataset = tf.data.Dataset.list_files(recoded_dir+os.sep+'*.jpg', seed=10000) # This also shuffles the images
    tamucc_dataset = tamucc_dataset.map(read_image_and_label)
    tamucc_dataset = tamucc_dataset.map(resize_and_crop_image, num_parallel_calls=AUTO)

    tamucc_dataset = tamucc_dataset.map(recompress_image, num_parallel_calls=AUTO)
    tamucc_dataset = tamucc_dataset.batch(shared_size)
    return tamucc_dataset

#-----------------------------------
def write_records(tamucc_dataset, tfrecord_dir, CLASSES):
    """
    "write_records"
    This function writes a tf.data.Dataset object to TFRecord shards
    INPUTS:
        * tamucc_dataset [tf.data.Dataset]
        * tfrecord_dir [string] : path to directory where files will be written
        * CLASSES [list] of class string names
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (files written to disk)
    """
    for shard, (image, label) in enumerate(tamucc_dataset):
      shard_size = image.numpy().shape[0]
      filename = tfrecord_dir+os.sep+"tamucc" + "{:02d}-{}.tfrec".format(shard, shard_size)

      with tf.io.TFRecordWriter(filename) as out_file:
        for i in range(shard_size):
          example = to_tfrecord(image.numpy()[i],label.numpy()[i], CLASSES)
          out_file.write(example.SerializeToString())
        print("Wrote file {} containing {} records".format(filename, shard_size))

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
