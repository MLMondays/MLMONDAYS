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


from oyster_imports import *

import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

##calcs
import tensorflow as tf #numerical operations on gpu
import numpy as np
import tensorflow.keras.backend as K

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

###############################################################
### TFRECORD FUNCTIONS
###############################################################

# this function annotation is to suppress warnings related toi use of conditional opeartors
@tf.autograph.experimental.do_not_convert
#-----------------------------------
def read_seg_tfrecord_obx_binary(example):
    """
    "read_seg_tfrecord_obx_binary(example)"
    This function reads an example from a TFrecord file into a single image and label
    In this binary image creator for OBX, input 4-class imagery is binarized based on
    0=63=deep, 1=128=broken, 1=191=shallow, 1=255=dry
    INPUTS:
        * TFRecord example object
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor array]
        * class_label [tensor array]
    """

    #list of features to extract from each tfrecord example
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "label": tf.io.FixedLenFeature([], tf.string),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32)/ 255.0
    image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])
    ##uncomment for greyscale imagery
    #image = tf.reshape(tf.image.rgb_to_grayscale(image), [TARGET_SIZE,TARGET_SIZE, 1])

    label = tf.image.decode_jpeg(example['label'], channels=1)
    label = tf.cast(label, tf.uint8)#/ 255.0
    label = tf.reshape(label, [TARGET_SIZE,TARGET_SIZE, 1])

    # #63=deep, 128=broken, 191=shallow, 255=dry
    for counter, val in enumerate([63,128,191,255]):
       cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*val)
       label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*counter, label)
       for k in range(1,10):
          cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*val-k)
          label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*counter, label)

    for counter, val in enumerate([63,128,191]): #cant go higher than 255
       for k in range(1,10):
          cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*val+k)
          label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*counter, label)

    cond = tf.greater(label, tf.ones(tf.shape(label),dtype=tf.uint8)*counter)
    label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*counter+1, label)

    # 0=63=deep, 1=128=broken, 2=191=shallow, 3=255=dry
    # make binary by >0 = 2, then {1} = 0, then {2} = 0
    # cond = tf.greater(label, tf.ones(tf.shape(label),dtype=tf.uint8)*0)
    # label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*4, label)
    cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*1)
    label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*0, label)
    cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*2)
    label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*0, label)
    cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*3)
    label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*1, label)
    #image = tf.image.per_image_standardization(image)

    return image, label


@tf.autograph.experimental.do_not_convert
#-----------------------------------
def read_seg_tfrecord_obx_multiclass(example):
    """
    "read_seg_tfrecord_obx_multiclass(example)"
    This function reads an example from a TFrecord file into a single image and label
    This is the "multiclass" version for OBS imagery, where the classes are mapped as follows:
    0=63=deep, 2=128=broken, 3=191=shallow, 4=255=dry
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
    #image = tf.reshape(tf.image.rgb_to_grayscale(image), [TARGET_SIZE,TARGET_SIZE, 1])

    label = tf.image.decode_jpeg(example['label'], channels=1)
    label = tf.cast(label, tf.uint8)#/ 255.0
    label = tf.reshape(label, [TARGET_SIZE,TARGET_SIZE, 1])

    #63 = deep, 255 = dry
    for counter, val in enumerate([63,128,191,255]):
       cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*val)
       label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*counter, label)
       for k in range(1,10):
          cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*val-k)
          label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*counter, label)

    for counter, val in enumerate([63,128,191]): #cant go higher than 255
       for k in range(1,10):
          cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*val+k)
          label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*counter, label)

    cond = tf.greater(label, tf.ones(tf.shape(label),dtype=tf.uint8)*counter)
    label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*counter+1, label)

    label = tf.one_hot(tf.cast(label, tf.uint8), 4)

    label = tf.squeeze(label)

    image = tf.reshape(image, (image.shape[0], image.shape[1], image.shape[2]))

    #image = tf.image.per_image_standardization(image)
    return image, label


#-----------------------------------
def get_batched_dataset_oysternet(filenames):
    """
    "get_batched_dataset_oysternet(filenames)"
    This function defines a workflow for the model to read data from
    tfrecord files by defining the degree of parallelism, batch size, pre-fetching, etc
    and also formats the imagery properly for model training
    (assumes oysternet by using read_seg_tfrecord_oysternet)
    INPUTS:
        * filenames [list]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: BATCH_SIZE, AUTO
    OUTPUTS: tf.data.Dataset object
    """
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = True

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    dataset = dataset.map(read_seg_tfrecord_oysternet, num_parallel_calls=AUTO)
    dataset = dataset.cache() # This dataset fits in RAM
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(AUTO) #

    return dataset

#-----------------------------------
def get_batched_dataset_obx(filenames, flag):
    """
    "get_batched_dataset_obx(filenames, flag)"
    This function defines a workflow for the model to read data from
    tfrecord files by defining the degree of parallelism, batch size, pre-fetching, etc
    and also formats the imagery properly for model training
    If input flag is 'binary', read_seg_tfrecord_obx_binary is used to read tfrecords
    and parse into two categories (deep vs everything else)
    If input flag is 'multiclass', read_seg_tfrecord_obx_multiclass is used to parse
    tfrecords into 4 classes,. recoded 0 through 3
    INPUTS:
        * filenames [list]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: BATCH_SIZE, AUTO
    OUTPUTS: tf.data.Dataset object
    """
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = True

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    if flag is 'binary':
        dataset = dataset.map(read_seg_tfrecord_obx_binary, num_parallel_calls=AUTO)
    else:
        dataset = dataset.map(read_seg_tfrecord_obx_multiclass, num_parallel_calls=AUTO)

    dataset = dataset.cache() # This dataset fits in RAM
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(AUTO) #

    return dataset

#-----------------------------------
def get_seg_dataset_for_tfrecords_oysternet(imdir, lab_path, shared_size):
    """
    "get_seg_dataset_for_tfrecords_oysternet(imdir, lab_path, shared_size)"
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
def get_seg_dataset_for_tfrecords_obx(imdir, lab_path, shared_size):
    """
    "get_seg_dataset_for_tfrecords_obx"
    This function reads an image and label and decodes both jpegs
    into bytestring arrays.
    This is the version for OBX data, which differs in use of both
    resize_and_crop_seg_image_obx and resize_and_crop_seg_image_obx
    for image pre-processing
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
    dataset = dataset.map(read_seg_image_and_label_obx)
    dataset = dataset.map(resize_and_crop_seg_image_obx, num_parallel_calls=AUTO)
    dataset = dataset.map(recompress_seg_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(shared_size)
    return dataset

#-----------------------------------
def read_seg_image_and_label(img_path):
    """
    "read_seg_image_and_label(img_path)"
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
def read_seg_image_and_label_obx(img_path):
    """
    "read_seg_image_and_label_obx(img_path)"
    This function reads an image and label and decodes both jpegs
    into bytestring arrays.
    This works by parsing out the label image filename from its image pair
    Thre are different rules for non-augmented versus augmented imagery
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
    lab_path = tf.strings.regex_replace(lab_path, ".jpg", "_deep_whitewater_shallow_no_water_label.jpg")
    lab_path = tf.strings.regex_replace(lab_path, "augimage", "auglabel")
    bits = tf.io.read_file(lab_path)
    label = tf.image.decode_jpeg(bits)

    return image, label


#-----------------------------------
def resize_and_crop_seg_image(image, label):
    """
    "resize_and_crop_seg_image(image, label)"
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
def resize_and_crop_seg_image_obx(image, label):
    """
    "resize_and_crop_seg_image_obx"
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

    # #63=deep, 128=broken, 191=shallow, 255=dry
    # for counter, val in enumerate([63,128,191,255]):
    #    cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*val)
    #    label = tf.where(cond, tf.ones(tf.shape(label),dtype=tf.uint8)*counter, label)
    # #
    # # cond = tf.equal(label, tf.ones(tf.shape(label),dtype=tf.uint8)*63)
    # # label = tf.where(cond,  tf.ones(tf.shape(label),dtype=tf.uint8)*0, label)

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
def write_seg_records_obx(dataset, tfrecord_dir):
    """
    "write_seg_records_obx(dataset, tfrecord_dir)"
    This function writes a tf.data.Dataset object to TFRecord shards
    The version for OBX data preprends "obx" to the filenames, but otherwise is identical
    to write_seg_records
    INPUTS:
        * dataset [tf.data.Dataset]
        * tfrecord_dir [string] : path to directory where files will be written
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (files written to disk)
    """
    for shard, (image, label) in enumerate(dataset):
      shard_size = image.numpy().shape[0]
      filename = tfrecord_dir+os.sep+"obx" + "{:02d}-{}.tfrec".format(shard, shard_size)

      with tf.io.TFRecordWriter(filename) as out_file:
        for i in range(shard_size):
          example = to_seg_tfrecord(image.numpy()[i],label.numpy()[i])
          out_file.write(example.SerializeToString())
        print("Wrote file {} containing {} records".format(filename, shard_size))

#-----------------------------------
def write_seg_records_oysternet(dataset, tfrecord_dir, filestr):
    """
    "write_seg_records_oysternet(dataset, tfrecord_dir, filestr)"
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
      filename = tfrecord_dir+os.sep+filestr + "{:02d}-{}.tfrec".format(shard, shard_size)

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
def read_seg_tfrecord_oysternet(example):
    """
    "read_seg_tfrecord_oysternet(example)"
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

    image = tf.image.adjust_contrast(image, 2)
    # image = tf.image.per_image_standardization(image)

    label = tf.image.decode_jpeg(example['label'], channels=1)
    label = tf.reshape(label, [TARGET_SIZE,TARGET_SIZE, 1])
    label = tf.cast(label, tf.float32)/ 255.0
    # label = tf.cast(label, tf.int32)

    return image, label


#-----------------------------------
def seg_file2tensor(f):
    """
    "seg_file2tensor(f)"
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
