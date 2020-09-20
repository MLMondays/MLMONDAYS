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

## OYSTER
from oyster_imports import *

##UNCOMMENT BELOW TO USE NWPU DATA
# from nwpu_imports import *

#see mlmondays blog post:
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

##calcs
import tensorflow as tf #numerical operations on gpu
import numpy as np
import tensorflow.keras.backend as K

#plots
import matplotlib.pyplot as plt

#utils
#keras functions for early stopping and model weights saving
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

###############################################################
### TFRECORD FUNCTIONS
###############################################################

#-----------------------------------
def get_seg_dataset_for_tfrecords(imdir, lab_path, shared_size):
    """
    "get_seg_dataset_for_tfrecords"
    This function reads an image and label and decodes both jpegs
    into bytestring arrays.
    This works because the images and labels have the same name
    but different paths, hence `tf.strings.regex_replace(img_path, "images", "labels")`
    INPUTS:
        * image [tensor array]
        * label [tensor array]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * label [tensor array]
    """
    dataset = tf.data.Dataset.list_files(imdir+os.sep+'*.jpg', seed=10000) # This also shuffles the images
    dataset = dataset.map(read_seg_image_and_label)
    dataset = dataset.map(resize_and_crop_seg_image, num_parallel_calls=AUTO)
    dataset = dataset.map(recompress_seg_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(shared_size)
    return dataset

#-----------------------------------
def read_seg_image_and_label(img_path):
    """
    "read_seg_image_and_label"
    This function reads an image and label and decodes both jpegs
    into bytestring arrays.
    This works because the images and labels have the same name
    but different paths, hence `tf.strings.regex_replace(img_path, "images", "labels")`
    INPUTS:
        * img_path [tensor string]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * image [bytestring]
        * label [bytestring]
    """
    bits = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(bits)

    # have to use this tf.strings.regex_replace utility because img_path is a Tensor object
    lab_path = tf.strings.regex_replace(img_path, "images", "labels")
    bits = tf.io.read_file(lab_path)
    label = tf.image.decode_jpeg(bits)

    return image, label

#-----------------------------------
def resize_and_crop_seg_image(image, label):
    """
    "resize_and_crop_seg_image"
    This function crops to square and resizes an image and label
    INPUTS:
        * image [tensor array]
        * label [tensor array]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * label [tensor array]
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

    label = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(label, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(label, [w*th/h, h*th/h])  # if false
                 )
    label = tf.image.crop_to_bounding_box(label, (nw - tw) // 2, (nh - th) // 2, tw, th)

    return image, label

#-----------------------------------
def recompress_seg_image(image, label):
    """
    "recompress_seg_image"
    This function takes an image and label encoded as a byte string
    and recodes as an 8-bit jpeg
    INPUTS:
        * image [tensor array]
        * label [tensor array]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * image [tensor array]
        * label [tensor array]
    """
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)

    label = tf.cast(label, tf.uint8)
    label = tf.image.encode_jpeg(label, optimize_size=True, chroma_downsampling=False)
    return image, label

#-----------------------------------
def write_seg_records(dataset, tfrecord_dir):
    """
    "write_records"
    This function writes a tf.data.Dataset object to TFRecord shards
    INPUTS:
        * dataset [tf.data.Dataset]
        * tfrecord_dir [string] : path to directory where files will be written
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (files written to disk)
    """
    for shard, (image, label) in enumerate(dataset):
      shard_size = image.numpy().shape[0]
      filename = tfrecord_dir+os.sep+"oysternet" + "{:02d}-{}.tfrec".format(shard, shard_size)

      with tf.io.TFRecordWriter(filename) as out_file:
        for i in range(shard_size):
          example = to_seg_tfrecord(image.numpy()[i],label.numpy()[i])
          out_file.write(example.SerializeToString())
        print("Wrote file {} containing {} records".format(filename, shard_size))

#-----------------------------------
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

#-----------------------------------
def to_seg_tfrecord(img_bytes, label_bytes):
    """
    "to_seg_tfrecord"
    This function creates a TFRecord example from an image byte string and a label feature
    INPUTS:
        * img_bytes
        * label_bytes
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: tf.train.Feature example
    """
    feature = {
      "image": _bytestring_feature([img_bytes]), # one image in the list
      "label": _bytestring_feature([label_bytes]), # one label image in the list
              }
    return tf.train.Example(features=tf.train.Features(feature=feature))

#-----------------------------------
def read_seg_tfrecord(example):
    """
    "read_seg_tfrecord"
    This function reads an example from a TFrecord file into a single image and label
    INPUTS:
        * TFRecord example object
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * class_label [tensor array]
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "label": tf.io.FixedLenFeature([], tf.string),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32)/ 255.0
    image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])

    label = tf.image.decode_jpeg(example['label'], channels=1)
    label = tf.cast(label, tf.float32)/ 255.0
    label = tf.reshape(label, [TARGET_SIZE,TARGET_SIZE, 1])
    #class_label = tf.cast(example['class'], tf.int32)

    return image, label

###############################################################
### MODEL FUNCTIONS
###############################################################

#-----------------------------------
def dice_coef(y_true, y_pred):
    """
    "dice_coef"
    This function
    INPUTS:
        *
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        *
        *
    """
    smooth = 1.
    y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

#-----------------------------------
def batchnorm_act(x):
    """
    "batchnorm_act"
    This function
    INPUTS:
        *
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        *
        *
    """
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("relu")(x)

#-----------------------------------
def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    """
    "conv_block"
    This function
    INPUTS:
        *
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        *
        *
    """
    conv = batchnorm_act(x)
    return tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)

#-----------------------------------
def bottleneck_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    """
    "bottleneck_block"
    This function
    INPUTS:
        *
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        *
        *
    """
    conv = tf.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    bottleneck = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    bottleneck = batchnorm_act(bottleneck)

    return tf.keras.layers.Add()([conv, bottleneck])

#-----------------------------------
def res_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    """
    "res_block"
    This function
    INPUTS:
        *
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        *
        *
    """
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    bottleneck = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    bottleneck = batchnorm_act(bottleneck)

    return tf.keras.layers.Add()([bottleneck, res])

#-----------------------------------
def upsamp_concat_block(x, xskip):
    """
    "upsamp_concat_block"
    This function
    INPUTS:
        *
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        *
        *
    """
    u = tf.keras.layers.UpSampling2D((2, 2))(x)
    return tf.keras.layers.Concatenate()([u, xskip])

#-----------------------------------
def res_unet(sz, f):
    """
    "res_unet"
    This function
    INPUTS:
        *
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        *
        *
    """
    inputs = tf.keras.layers.Input(sz)

    ## downsample
    e1 = bottleneck_block(inputs, f); f = int(f*2)
    e2 = res_block(e1, f, strides=2); f = int(f*2)
    e3 = res_block(e2, f, strides=2); f = int(f*2)
    e4 = res_block(e3, f, strides=2); f = int(f*2)
    _ = res_block(e4, f, strides=2)

    ## bottleneck
    b0 = conv_block(_, f, strides=1)
    _ = conv_block(b0, f, strides=1)

    ## upsample
    _ = upsamp_concat_block(_, e4)
    _ = res_block(_, f); f = int(f/2)

    _ = upsamp_concat_block(_, e3)
    _ = res_block(_, f); f = int(f/2)

    _ = upsamp_concat_block(_, e2)
    _ = res_block(_, f); f = int(f/2)

    _ = upsamp_concat_block(_, e1)
    _ = res_block(_, f)

    ## classify
    outputs = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(_)

    #model creation
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

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


#-----------------------------------
def seg_file2tensor(f):
    """
    "file2tensor"
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained segmentation model
    INPUTS:
        * f [string] file name of jpeg
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]: unstandardized image
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
    # image = tf.cast(image, tf.uint8) #/ 255.0

    return image

###############################################################
### PLOTTING FUNCTIONS
###############################################################

#-----------------------------------
def plot_seg_history(history, train_hist_fig):
    """
    "plot_seg_history"
    This function plots the training history of a model
    INPUTS:
        * history [dict]: the output dictionary of the model.fit() process, i.e. history = model.fit(...)
        * train_hist_fig [string]: the filename where the plot will be printed
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (figure printed to file)
    """
    n = len(history.history['val_loss'])

    plt.figure(figsize=(20,10))
    plt.subplot(121)
    plt.plot(np.arange(1,n+1), history.history['dice_coef'], 'b', label='train accuracy')
    plt.plot(np.arange(1,n+1), history.history['val_dice_coef'], 'k', label='validation accuracy')
    plt.xlabel('Epoch number', fontsize=10); plt.ylabel('Dice Coefficient', fontsize=10)
    plt.legend(fontsize=10)

    plt.subplot(122)
    plt.plot(np.arange(1,n+1), history.history['loss'], 'b', label='train loss')
    plt.plot(np.arange(1,n+1), history.history['val_loss'], 'k', label='validation loss')
    plt.xlabel('Epoch number', fontsize=10); plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=10)

    # plt.show()
    plt.savefig(train_hist_fig, dpi=200, bbox_inches='tight')


def make_sample_seg_plot(model, sample_filenames, test_samples_fig):
    """
    "make_sample_seg_plot"
    This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
    using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
    saving out to the provided filename, cm_filename
    INPUTS:
        * model: trained and compiled keras model
        * sample_filenames: [list] of strings
        * test_samples_fig [string]: filename to print figure to
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (matplotlib figure, printed to file)
    """

    plt.figure(figsize=(16,16))

    for counter,f in enumerate(sample_filenames):
        image = seg_file2tensor(f)/255
        est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
        est_label[est_label>0.5] = 1
        est_label = (est_label*255).astype(np.uint8)

        plt.subplot(6,4,counter+1)
        name = sample_filenames[counter].split(os.sep)[-1].split('_')[0]
        plt.title(name, fontsize=10)
        plt.imshow(image)
        plt.imshow(est_label, alpha=0.5, cmap=plt.cm.gray)

        plt.axis('off')

    # plt.show()
    plt.savefig(test_samples_fig,
                dpi=200, bbox_inches='tight')
    plt.close('all')
