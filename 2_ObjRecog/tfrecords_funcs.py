
from coco_imports import *

#see mlmondays blog post:
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

##calcs
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
import io
from collections import namedtuple
from data_funcs import resize_and_pad_image, LabelEncoderCoco, preprocess_coco_data, preprocess_secoora_data

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)


###===============================================================
def compute_iou(boxes1, boxes2):
    """
    compute_iou(boxes1, boxes2)
    This function computes pairwise IOU matrix for given two sets of boxes
    INPUTS:
        * boxes1: A tensor with shape `(N, 4)` representing bounding boxes
          where each box is of the format `[x, y, width, height]`.
        * boxes2: A tensor with shape `(M, 4)` representing bounding boxes
          where each box is of the format `[x, y, width, height]`.
    OPTIONAL INPUTS: None
    OUTPUTS:
        *  pairwise IOU matrix with shape `(N, M)`, where the value at ith row
           jth column holds the IOU between ith box and jth box from
           boxes1 and boxes2 respectively.
    GLOBAL INPUTS: None
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


class AnchorBox:
    """
    "AnchorBox"
    ## Code from https://keras.io/examples/vision/retinanet/
    Generates anchor boxes.
    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.
    INPUTS:
      * aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      * scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      * num_anchors: The number of anchor boxes at each location on feature map
      * areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      * strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    OPTIONAL INPUTS: None
    OUTPUTS:
        * anchor boxes for all the feature maps, stacked as a single tensor with shape
        `(total_anchors, 4)`, when AnchorBox._get_anchors() is called
    GLOBAL INPUTS: None
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """
        "_get_anchors"
        ## Code from https://keras.io/examples/vision/retinanet/
        Generates anchor boxes for a given feature map size and level
        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.
        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """
        "get_anchors"
        ## Code from https://keras.io/examples/vision/retinanet/
        Generates anchor boxes for all the feature maps of the feature pyramid.
        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.
        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)

###############################################################
## TFRECORDS
###############################################################
def prepare_image(image):
    """
    prepare_image(image)
    ""
    This function resizes and pads an image, and rescales for resnet
    INPUTS:
        * image [tensor array]
    OPTIONAL INPUTS: None
    OUTPUTS:
        * image [tensor array]
    GLOBAL INPUTS: None
    """
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio

#----------------------------------------------
def get_test_secoora_dataset(val_filenames):
    """
    prepare_secoora_datasets_for_training(data_path, val_filenames):
    This funcion prepares train and validation datasets  by extracting features (images, bounding boxes, and class labels)
    then map to preprocess_secoora_data, then apply prefetch, padded batch and label encoder
    INPUTS:
        * data_path [string]: path to the tfrecords
        * train_filenames [string]: tfrecord filenames for training
        * val_filenames [string]: tfrecord filenames for validation
    OPTIONAL INPUTS: None
    OUTPUTS:
        * val_dataset [tensorflow dataset]: validation dataset
        * train_dataset [tensorflow dataset]: training dataset
    GLOBAL INPUTS: None
    """

    features = {
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'objects/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'objects/ymin': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'objects/xmax': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'objects/ymax': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'objects/label': tf.io.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
    }

    def _parse_function(example_proto):
      # Parse the input `tf.train.Example` proto using the dictionary above.
      return tf.io.parse_single_example(example_proto, features)

    dataset = tf.data.TFRecordDataset(val_filenames)
    dataset = dataset.map(_parse_function)
    return dataset
    
#----------------------------------------------
def prepare_secoora_datasets_for_training(data_path, train_filenames, val_filenames):
    """
    prepare_secoora_datasets_for_training(data_path, train_filenames, val_filenames):
    This funcion prepares train and validation datasets  by extracting features (images, bounding boxes, and class labels)
    then map to preprocess_secoora_data, then apply prefetch, padded batch and label encoder
    INPUTS:
        * data_path [string]: path to the tfrecords
        * train_filenames [string]: tfrecord filenames for training
        * val_filenames [string]: tfrecord filenames for validation
    OPTIONAL INPUTS: None
    OUTPUTS:
        * val_dataset [tensorflow dataset]: validation dataset
        * train_dataset [tensorflow dataset]: training dataset
    GLOBAL INPUTS: None
    """

    features = {
        'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'objects/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'objects/ymin': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'objects/xmax': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'objects/ymax': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
        'objects/label': tf.io.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
    }

    def _parse_function(example_proto):
      # Parse the input `tf.train.Example` proto using the dictionary above.
      return tf.io.parse_single_example(example_proto, features)

    train_dataset = tf.data.TFRecordDataset(train_filenames)
    train_dataset = train_dataset.map(_parse_function)

    train_dataset = train_dataset.map(preprocess_secoora_data, num_parallel_calls=AUTO)

    shapes = (tf.TensorShape([None,None,3]),tf.TensorShape([None,4]),tf.TensorShape([None,]))

    # this is necessary because there are unequal numbers of labels in every image
    train_dataset = train_dataset.padded_batch(
        batch_size = BATCH_SIZE, drop_remainder=True, padding_values=(0.0, 1e-8, -1), padded_shapes=shapes,
    )

    # i dont understand this!!
    label_encoder = LabelEncoderCoco()

    # train_dataset = train_dataset.shuffle(8 * BATCH_SIZE)
    train_dataset = train_dataset.map(
        label_encoder.encode_batch, num_parallel_calls=AUTO
    )

    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(AUTO)

    val_dataset = tf.data.TFRecordDataset(val_filenames)
    val_dataset = val_dataset.map(_parse_function)
    val_dataset = val_dataset.map(preprocess_secoora_data, num_parallel_calls=AUTO)

    val_dataset = val_dataset.padded_batch(
        batch_size = BATCH_SIZE, padding_values=(0.0, 1e-8, -1), drop_remainder=True, padded_shapes=shapes,
    )

    val_dataset = val_dataset.map(
        label_encoder.encode_batch, num_parallel_calls=AUTO
    )
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    val_dataset = val_dataset.prefetch(AUTO)

    return train_dataset, val_dataset


#----------------------------------------------
def prepare_coco_datasets_for_training(train_dataset, val_dataset):
    """
    prepare_coco_datasets_for_training(train_dataset, val_dataset)
    This function prepares a coco dataset loaded from tfds into one trainable by the model
    INPUTS:
        * val_dataset [tensorflow dataset]: validation dataset
        * train_dataset [tensorflow dataset]: training dataset
    OPTIONAL INPUTS: None
    OUTPUTS:
        * val_dataset [tensorflow dataset]: validation dataset
        * train_dataset [tensorflow dataset]: training dataset
    GLOBAL INPUTS: BATCH_SIZE
    """
    # ## Encoding labels
    # The raw labels, consisting of bounding boxes and class ids need to be
    # transformed into targets for training. This transformation consists of
    # the following steps:
    # - Generating anchor boxes for the given image dimensions
    # - Assigning ground truth boxes to the anchor boxes
    # - The anchor boxes that are not assigned any objects, are either assigned the
    # background class or ignored depending on the IOU
    # - Generating the classification and regression targets using anchor boxes

    label_encoder = LabelEncoderCoco()

    train_dataset = train_dataset.map(preprocess_coco_data, num_parallel_calls=AUTO)

    train_dataset = train_dataset.shuffle(8 * BATCH_SIZE)
    train_dataset = train_dataset.padded_batch(
        batch_size = BATCH_SIZE, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )

    train_dataset = train_dataset.map(
        label_encoder.encode_batch, num_parallel_calls=AUTO
    )
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(AUTO)


    val_dataset = val_dataset.map(preprocess_coco_data, num_parallel_calls=AUTO)
    val_dataset = val_dataset.padded_batch(
        batch_size = BATCH_SIZE, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=AUTO)
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    val_dataset = val_dataset.prefetch(AUTO)
    return train_dataset, val_dataset

#-----------------------------------
def file2tensor(f):
    """
    file2tensor(f)
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

    return image

#------------------------------------------------------
def write_tfrecords(output_path, image_dir, csv_input):
    """
    write_tfrecords(output_path, image_dir, csv_input)
    ""
    This function writes tfrecords to fisk
    INPUTS:
        * image_dir [string]: place where jpeg images are
        * csv_input [string]: csv file that contains the labels
        * output_path [string]: place to writes files to
    OPTIONAL INPUTS: None
    OUTPUTS: None (tfrecord files written to disk)
    GLOBAL INPUTS: BATCH_SIZE
    """
    writer = tf.io.TFRecordWriter(output_path)

    path = os.path.join(os.getcwd(),image_dir)

    examples = pd.read_csv(csv_input)
    print(len(examples))
    grouped = split(examples, 'filename')

    for group in grouped:
        tf_example = create_tf_example_coco(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

#------------------------------------------------------
def class_text_to_int(row_label):
    """
    class_text_to_int(row_label)
    ""
    This function converts the string 'person' into the number 1
    INPUTS:
        * row_label [string]: class label string
    OPTIONAL INPUTS: None
    OUTPUTS: 1 or None
    GLOBAL INPUTS: BATCH_SIZE
    """
    if row_label == 'person':
        return 1
    else:
        None

def split(df, group):
    """
    split(df, group)
    ""
    This function splits a pandas dataframe by a pandas group object
    to extract the label sets from each image
    for writing to tfrecords
    INPUTS:
        * df [pandas dataframe]
        * group [pandas dataframe group object]
    OPTIONAL INPUTS: None
    OUTPUTS:
        * tuple of bboxes and classes per image
    GLOBAL INPUTS: BATCH_SIZE
    """
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example_coco(group, path):
    """
    create_tf_example_coco(group, path
    ""
    This function creates an example tfrecord consisting of an image and label encoded as bytestrings
    The jpeg image is read into a bytestring, and the bbox coordinates and classes are collated and
    converted also
    INPUTS:
        * group [pandas dataframe group object]
        * path [tensorflow dataset]: training dataset
    OPTIONAL INPUTS: None
    OUTPUTS:
        * tf_example [tf.train.Example object]
    GLOBAL INPUTS: BATCH_SIZE
    """
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)

    filename = group.filename.encode('utf8')
    # image_format = b'jpg'

    ids = []
    areas = []
    xmins = [] ; xmaxs = []; ymins = []; ymaxs = []
    labels = []
    is_crowds = []

    for index, row in group.object.iterrows():
        labels.append(class_text_to_int(row['class']))
        ids.append(index)
        xmins.append(row['xmin'])
        ymins.append(row['ymin'])
        xmaxs.append(row['xmax'])
        ymaxs.append(row['ymax'])
        areas.append((row['xmax']-row['xmin'])*(row['ymax']-row['ymin']))
        is_crowds.append(False)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'objects/is_crowd': int64_list_feature(is_crowds),
        'image/filename': bytes_feature(filename),
        'image/id': int64_list_feature(ids),
        'image': bytes_feature(encoded_jpg),
        'objects/xmin': float_list_feature(xmins), #xs
        'objects/xmax': float_list_feature(xmaxs), #xs
        'objects/ymin': float_list_feature(ymins), #xs
        'objects/ymax': float_list_feature(ymaxs), #xs
        'objects/area': float_list_feature(areas), #ys
        'objects/id': int64_list_feature(ids), #ys
        'objects/label': int64_list_feature(labels),
    }))

    return tf_example
