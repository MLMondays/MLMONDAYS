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
import os, json
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf #numerical operations on gpu
import numpy as np #numerical operations on cpu

# set a seed for reproducibility
SEED=42
np.random.seed(SEED)
tf.random.set_seed(SEED)

#for automatically determining dataset feeding processes based on available hardware
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

###############################################################
### TFRECORD FUNCTIONS
###############################################################

#-----------------------------------
def read_classes_from_json(json_file):
    """
    read_classes_from_json(json_file)
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

#-----------------------------------
def file2tensor(f, model='mobilenet'):
    """
    file2tensor(f, model='mobilenet')
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained mobilenet or vgg model
    (the imagery is standardized depedning on target model framework)
    INPUTS:
        * f [string] file name of jpeg
    OPTIONAL INPUTS:
        * model = {'mobilenet' | 'vgg'}
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
    image = tf.keras.preprocessing.image.load_img(f, target_size=(TARGET_SIZE, TARGET_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)

    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    # image = tf.cast(image, tf.uint8) #/ 255.0
    if model == 'mobilenet':
       im = tf.keras.applications.mobilenet_v2.preprocess_input(image) #specific to mobilenetV2
    elif model=='vgg':
       im = tf.keras.applications.vgg16.preprocess_input(image) #specific to vgg16

    return image, im

#-----------------------------------
def get_batched_dataset(filenames):
    """
    get_batched_dataset(filenames)
    This function defines a workflow for the model to read data from
    tfrecord files by defining the degree of parallelism, batch size, pre-fetching, etc
    and also formats the imagery properly for model training
    (assumes mobilenet by using read_tfrecord_mv2)
    INPUTS:
        * filenames [list]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: BATCH_SIZE, AUTO
    OUTPUTS: tf.data.Dataset object
    """
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = True # False

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    dataset = dataset.map(read_tfrecord_mv2, num_parallel_calls=AUTO)

    dataset = dataset.cache() # This dataset fits in RAM
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(AUTO) #

    return dataset

#-----------------------------------
def get_eval_dataset(filenames):
    """
    get_eval_dataset(filenames)
    This function defines a workflow for the model to read data from
    tfrecord files by defining the degree of parallelism, batch size, pre-fetching, etc
    and also formats the imagery properly for model training
    (assumes mobilenet by using read_tfrecord_mv2)

    This evaluation version does not .repeat() because it is not being called repeatedly by a model
    INPUTS:
        * filenames [list]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: BATCH_SIZE, AUTO
    OUTPUTS: tf.data.Dataset object
    """
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = True #False

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=BATCH_SIZE, num_parallel_calls=AUTO)
    dataset = dataset.map(read_tfrecord_mv2, num_parallel_calls=AUTO)

    dataset = dataset.cache() # This dataset fits in RAM
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(AUTO) #

    return dataset


#-----------------------------------
def read_tfrecord_vgg(example):
    """
    read_tfrecord_vgg(example)
    This function reads an example record from a tfrecord file
    and parses into label and image ready for vgg model training
    INPUTS:
        * example: an tfrecord 'example' object, containing an image and label
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor]: resized and pre-processed for vgg
        * class_label [tensor] 32-bit integer
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) #/ 255.0

    image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])
    #image = tf.image.per_image_standardization(image)
    image = tf.keras.applications.vgg.preprocess_input(image) #specific to model

    class_label = tf.cast(example['class'], tf.int32)

    return image, class_label


#-----------------------------------
def read_tfrecord_mv2(example):
    """
    read_tfrecord_mv2(example)
    This function reads an example record from a tfrecord file
    and parses into label and image ready for mobilenet model training
    INPUTS:
        * example: an tfrecord 'example' object, containing an image and label
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor]: resized and pre-processed for mobilenetv2
        * class_label [tensor] 32-bit integer
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) #/ 255.0

    image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])
    #image = tf.image.per_image_standardization(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image) #specific to model

    class_label = tf.cast(example['class'], tf.int32)

    return image, class_label

#-----------------------------------
def resize_and_crop_image(image, label):
    """
    resize_and_crop_image(image, label)
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
    recompress_image(image, label)
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
    "_bytestring_feature(list_of_bytestrings)"
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
    "_int_feature(list_of_ints)"
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
    "_float_feature(list_of_floats)"
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
    to_tfrecord(img_bytes, label, CLASSES)
    This function creates a TFRecord example from an image byte string and a label feature
    INPUTS:
        * img_bytes: an image bytestring
        * label: label string of image
        * CLASSES: list of string classes in the entire dataset
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
    read_tfrecord(example)
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
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])

    class_label = tf.cast(example['class'], tf.int32)

    return image, class_label

#-----------------------------------
def read_image_and_label(img_path):
    """
    read_image_and_label(img_path)
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
    get_dataset_for_tfrecords(recoded_dir, shared_size)
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
    write_records(tamucc_dataset, tfrecord_dir, CLASSES)
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
