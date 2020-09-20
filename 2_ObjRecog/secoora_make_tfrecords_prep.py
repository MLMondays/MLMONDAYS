

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
# from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
import numpy as np

TARGET_SIZE = 512



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



# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'person':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'

    xs = []
    ys = []
    ws = []
    hs = []
    # classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        # classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
        #bbox = [(row['xmax']+row['xmin'])/2, (row['ymax']+row['ymin'])/2, row['xmax']-row['xmin'], row['ymax']-row['ymin']]
        xs.append((row['xmax']+row['xmin'])/2)
        ys.append((row['ymax']+row['ymin'])/2)
        ws.append(row['xmax']-row['xmin'])
        hs.append(row['ymax']-row['ymin'])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'height': int64_feature(height),
        'width': int64_feature(width),
        'filename': bytes_feature(filename),
        'id': bytes_feature(filename),
        'image': bytes_feature(encoded_jpg),
        'format': bytes_feature(image_format),
        'objects/bbox/xs': float_list_feature(xs),
        'objects/bbox/ys': float_list_feature(ys),
        'objects/bbox/ws': float_list_feature(ws),
        'objects/bbox/hs': float_list_feature(hs), #replace with a N, 4 matrix as bbox
        # 'objects/class/id': bytes_list_feature(classes_text),
        'objects/class/label': int64_list_feature(classes),
    }))
    return tf_example


# root = '/media/marda/TWOTB/USGS/SOFTWARE/mlmondays_prep/obj_recog/Pedestrian-Detection/images'+os.sep

root = '/media/marda/TWOTB/USGS/SOFTWARE/DL-CDI2020/2_ObjRecog/data/secoora'+os.sep

output_path = root+'secoora-train.tfrecord'

image_dir = root+'train'

csv_input = root+'train_labels.csv'

writer = tf.io.TFRecordWriter(output_path)

path = os.path.join(os.getcwd(),image_dir)

examples = pd.read_csv(csv_input)
grouped = split(examples, 'filename')

for group in grouped:
    tf_example = create_tf_example(group, path)
    writer.write(tf_example.SerializeToString())

writer.close()
output_path = os.path.join(os.getcwd(), output_path)
print('Successfully created the TFRecords: {}'.format(output_path))


##================


output_path = root+'secoora-test.tfrecord'

image_dir = root+'test'

csv_input = root+'test_labels.csv'

writer = tf.io.TFRecordWriter(output_path)

path = os.path.join(os.getcwd(),image_dir)

examples = pd.read_csv(csv_input)
grouped = split(examples, 'filename')

for group in grouped:
    tf_example = create_tf_example(group, path)
    writer.write(tf_example.SerializeToString())

writer.close()
output_path = os.path.join(os.getcwd(), output_path)
print('Successfully created the TFRecords: {}'.format(output_path))


##================


output_path = root+'secoora-validation.tfrecord'

image_dir = root+'validation'

csv_input = root+'validation_labels.csv'

writer = tf.io.TFRecordWriter(output_path)

path = os.path.join(os.getcwd(),image_dir)

examples = pd.read_csv(csv_input)
grouped = split(examples, 'filename')

for group in grouped:
    tf_example = create_tf_example(group, path)
    writer.write(tf_example.SerializeToString())

writer.close()
output_path = os.path.join(os.getcwd(), output_path)
print('Successfully created the TFRecords: {}'.format(output_path))




def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, features)


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates
    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.
    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


def preprocess_data(example):
    """Applies preprocessing step to a single sample
    Arguments:
      sample: A dict representing a single training sample.
    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32)/ 255.0

    xs=tf.cast(example['objects/bbox/xs'], tf.int32)
    ys=tf.cast(example['objects/bbox/ys'], tf.int32)
    ws=tf.cast(example['objects/bbox/ws'], tf.int32)
    hs=tf.cast(example['objects/bbox/hs'], tf.int32)

    bbox = tf.reshape(tf.concat((xs,ys,ws,hs), axis=0), (-1,4))

    #print(tf.cast(example['objects/bbox/xs'], tf.float32))
    #bbox = np.c_[example['objects/bbox/xs'], example['objects/bbox/ys'],
                 # example['objects/bbox/ws'], example['objects/bbox/hs']]
    # bbox = tf.cast(bbox, tf.float32)
    # bbox = convert_to_xywh(bbox)

    # bbox = swap_xy(example["objects/bbox"])
    class_id = tf.cast(example["objects/class/label"], dtype=tf.int32)

    # image, bbox = random_flip_horizontal(image, bbox)
    # image, image_shape, _ = resize_and_pad_image(image)

    # bbox2 = tf.stack(
    #     [
    #         bbox[:, 0] * image_shape[1],
    #         bbox[:, 1] * image_shape[0],
    #         bbox[:, 2] * image_shape[1],
    #         bbox[:, 3] * image_shape[0],
    #     ],
    #     axis=-1,
    # )
    return image, bbox, class_id


data_path = '/media/marda/TWOTB/USGS/SOFTWARE/DL-CDI2020/2_ObjRecog/data/dummy_data/2020/0.0.0'

AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API
#
# filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrecord'))
# dataset = tf.data.Dataset.list_files(filenames)


features = {
    'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'format': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'objects/bbox/xs': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'objects/bbox/ys': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
    'objects/bbox/ws': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
    'objects/bbox/hs': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True), #replace with a N, 4 matrix as bbox
    'objects/class/id': tf.io.FixedLenSequenceFeature([], tf.string,allow_missing=True),
    'objects/class/label': tf.io.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
}

filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrecord'))
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
dataset = dataset.map(preprocess_data, num_parallel_calls=AUTO)


for i,b,c in dataset.take(1):
    ...:     print(b)



# for raw_record in raw_dataset.take(1):
#   print(repr(raw_record))




# for example in parsed_dataset.take(1):
#   print(repr(example))
#
# for example in parsed_dataset.take(1):
#   print(repr(example))
#
# TARGET_SIZE = 512
# image = tf.image.decode_jpeg(example['image'], channels=3)
# image = tf.cast(image, tf.float32)/ 255.0
# # image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])
#
# #
#
#
#     features = {
#         "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
#         "label": tf.io.FixedLenFeature([], tf.string),   # shape [] means scalar
#     }
#     # decode the TFRecord
#     example = tf.io.parse_single_example(example, features)
#
#     image = tf.image.decode_jpeg(example['image'], channels=3)
#     image = tf.cast(image, tf.float32)/ 255.0
#     image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])
#
#
#
#
#
# def read_seg_tfrecord(example):
#
#     features = {
#         'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#         'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#         'filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
#         'id': tf.io.FixedLenFeature([], tf.string, default_value=''),
#         'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
#         'format': tf.io.FixedLenFeature([], tf.string, default_value=''),
#         # 'image/objects/bbox/xmin': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#         # 'image/objects/bbox/xmax': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#         # 'image/objects/bbox/ymin': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
#         # 'image/objects/bbox/ymax': tf.io.FixedLenFeature([], tf.float32, default_value=0.0), #replace with a N, 4 matrix as bbox
#         # 'image/objects/class/id': tf.io.FixedLenFeature([], tf.string, default_value=''),
#         # 'image/objects/class/label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     }
#     # decode the TFRecord
#     example = tf.io.parse_single_example(example, features)
#     image = tf.image.decode_jpeg(example['image'], channels=3)
#     image = tf.cast(image, tf.float32)/ 255.0
#     image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])
#
#
#     # label = tf.image.decode_jpeg(example['label'], channels=1)
#     # label = tf.cast(label, tf.float32)/ 255.0
#     # label = tf.reshape(label, [TARGET_SIZE,TARGET_SIZE, 1])
#     #class_label = tf.cast(example['class'], tf.int32)
#     return image
#
#
# TARGET_SIZE = 512
# filenames = ['/media/marda/TWOTB/USGS/SOFTWARE/DL-CDI2020/2_ObjRecog/data/dummy_data/2020/0.0.0/dummy_data-train.tfrecord']
#
# option_no_order = tf.data.Options()
# option_no_order.experimental_deterministic = True
#
# AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API
#
# dataset = tf.data.Dataset.list_files(filenames)
# dataset = dataset.with_options(option_no_order)
# dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
# dataset = dataset.map(read_seg_tfrecord, num_parallel_calls=AUTO)
#
