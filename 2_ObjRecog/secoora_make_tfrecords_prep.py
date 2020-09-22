

from imports import *
#
#
# # TO-DO replace this with label map
# def class_text_to_int(row_label):
#     if row_label == 'person':
#         return 1
#     else:
#         None
#
#
# def split(df, group):
#     data = namedtuple('data', ['filename', 'object'])
#     gb = df.groupby(group)
#     return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
#
#

# def create_tf_example(group, path):
#     with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
#         encoded_jpg = fid.read()
#     encoded_jpg_io = io.BytesIO(encoded_jpg)
#     image = Image.open(encoded_jpg_io)
#     width, height = image.size
#
#     filename = group.filename.encode('utf8')
#     image_format = b'jpg'
#
#     xs = []
#     ys = []
#     ws = []
#     hs = []
#     classes = []
#     # xmins = []
#     # ymins = []
#     # xmaxs = []
#     # ymaxs = []
#
#     for index, row in group.object.iterrows():
#         # classes_text.append(row['class'].encode('utf8'))
#         classes.append(class_text_to_int(row['class']))
#         xs.append(row['xmin']) #(row['xmax']+row['xmin'])/2)
#         ys.append(row['ymin']) #(row['ymax']+row['ymin'])/2)
#         ws.append(row['xmax']-row['xmin'])
#         hs.append(row['ymax']-row['ymin'])
#         # xmins.append(row['xmin'])
#         # ymins.append(row['ymin'])
#         # xmaxs.append(row['xmax'])
#         # ymaxs.append(row['ymax'])
#
#     tf_example = tf.train.Example(features=tf.train.Features(feature={
#         'height': int64_feature(height),
#         'width': int64_feature(width),
#         'filename': bytes_feature(filename),
#         'id': bytes_feature(filename),
#         'image': bytes_feature(encoded_jpg),
#         'format': bytes_feature(image_format),
#         'objects/bbox/xs': float_list_feature(xs), #xs
#         'objects/bbox/ys': float_list_feature(ys), #ys
#         'objects/bbox/ws': float_list_feature(ws), #ws
#         'objects/bbox/hs': float_list_feature(hs), #hs
#         'objects/class/label': int64_list_feature(classes),
#     }))
#
#     # tf_example = tf.train.Example(features=tf.train.Features(feature={
#     #     'height': int64_feature(height),
#     #     'width': int64_feature(width),
#     #     'filename': bytes_feature(filename),
#     #     'id': bytes_feature(filename),
#     #     'image': bytes_feature(encoded_jpg),
#     #     'format': bytes_feature(image_format),
#     #     'objects/bbox/xmin': float_list_feature(xmins), #xs
#     #     'objects/bbox/ymin': float_list_feature(ymins), #ys
#     #     'objects/bbox/xmax': float_list_feature(xmaxs), #ws
#     #     'objects/bbox/ymax': float_list_feature(ymaxs), #hs
#     #     'objects/class/label': int64_list_feature(classes),
#     # }))
#     return tf_example

def create_tf_example_coco(group, path):
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


#
# # root = '/media/marda/TWOTB/USGS/SOFTWARE/mlmondays_prep/obj_recog/Pedestrian-Detection/images'+os.sep
#
# root = '/media/marda/TWOTB/USGS/SOFTWARE/DL-CDI2020/2_ObjRecog/data/secoora'+os.sep
#
# output_path = root+'secoora-train.tfrecord'
#
# image_dir = root+'train'
#
# csv_input = root+'train_labels.csv'
#
# writer = tf.io.TFRecordWriter(output_path)
#
# path = os.path.join(os.getcwd(),image_dir)
#
# examples = pd.read_csv(csv_input)
# grouped = split(examples, 'filename')
#
# for group in grouped:
#     tf_example = create_tf_example(group, path)
#     writer.write(tf_example.SerializeToString())
#
# writer.close()
# output_path = os.path.join(os.getcwd(), output_path)
# print('Successfully created the TFRecords: {}'.format(output_path))


##================

#
# output_path = root+'secoora-validation.tfrecord'
#
# image_dir = root+'validation'
#
# csv_input = root+'validation_labels.csv'
#
# writer = tf.io.TFRecordWriter(output_path)
#
# path = os.path.join(os.getcwd(),image_dir)
#
# examples = pd.read_csv(csv_input)
# grouped = split(examples, 'filename')
#
# for group in grouped:
#     tf_example = create_tf_example(group, path)
#     writer.write(tf_example.SerializeToString())
#
# writer.close()
# output_path = os.path.join(os.getcwd(), output_path)
# print('Successfully created the TFRecords: {}'.format(output_path))



data_path = "/media/marda/TWOTB/USGS/SOFTWARE/DL-CDI2020/2_ObjRecog/data/secoora"

AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

# features = {
#     'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'id': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'format': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'objects/bbox/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
#     'objects/bbox/ymin': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
#     'objects/bbox/xmax': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
#     'objects/bbox/ymax': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
#     'objects/class/id': tf.io.FixedLenSequenceFeature([], tf.string,allow_missing=True),
#     'objects/class/label': tf.io.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
# }

# features = {
#     'height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'id': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'format': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'objects/bbox/xs': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
#     'objects/bbox/ys': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
#     'objects/bbox/ws': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
#     'objects/bbox/hs': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
#     'objects/class/id': tf.io.FixedLenSequenceFeature([], tf.string,allow_missing=True),
#     'objects/class/label': tf.io.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
# }

features = {
    # 'filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
    # 'id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    # 'objects/bbox': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    'objects/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    'objects/ymin': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
    'objects/xmax': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
    'objects/ymax': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
    # 'objects/iscrowd': tf.io.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
    # 'objects/area': tf.io.FixedLenSequenceFeature([], tf.float32,allow_missing=True),
    # 'objects/id': tf.io.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
    'objects/label': tf.io.FixedLenSequenceFeature([], tf.int64,allow_missing=True),
}


def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, features)



#
BATCH_SIZE = 8


def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, features)



train_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*train*.tfrecord'))
train_dataset = tf.data.TFRecordDataset(train_filenames)
train_dataset = train_dataset.map(_parse_function)

train_dataset = train_dataset.map(preprocess_secoora_data, num_parallel_calls=AUTO)

# for a in train_dataset.take(1):
#     print(a)


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

# for a, b in train_dataset.take(1):
#     print(b)

train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(AUTO)



#
# train_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*train*.tfrecord'))
# train_dataset = tf.data.TFRecordDataset(train_filenames)
# train_dataset = train_dataset.map(_parse_function)
# train_dataset = train_dataset.map(preprocess_secoora_data, num_parallel_calls=AUTO)
#
# for a in train_dataset.take(1):
#     print(a)
#
#
# # this is necessary because there are unequal numbers of labels in every image
# train_dataset = train_dataset.padded_batch(
#     batch_size = BATCH_SIZE, drop_remainder=True, padding_values=(0.0, 1e-8, -1),
# )
#
# # for a in train_dataset.take(1):
# #     print(a)
#
# label_encoder = LabelEncoder()
#
# train_dataset = train_dataset.map(
#     label_encoder.encode_batch, num_parallel_calls=AUTO
# )
# train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
# train_dataset = train_dataset.prefetch(AUTO)
#
# # for a, b in train_dataset.take(1):
# #     print(a)
#
#
# val_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*val*.tfrecord'))
# val_dataset = tf.data.TFRecordDataset(val_filenames)
# val_dataset = val_dataset.map(_parse_function)
# val_dataset = val_dataset.map(preprocess_secoora_data, num_parallel_calls=AUTO)
#
# val_dataset = val_dataset.padded_batch(
#     batch_size = BATCH_SIZE, padding_values=(0.0, 1e-8, -1), drop_remainder=True
# )
#
# label_encoder = LabelEncoder()
#
# val_dataset = val_dataset.map(
#     label_encoder.encode_batch, num_parallel_calls=AUTO
# )
# val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
# val_dataset = val_dataset.prefetch(AUTO)
#
# for a, b in val_dataset.take(1):
#     print(a)

# a tuple composed of 3 elements: 0=image numpy float32, 1= <tf.Tensor: shape=(3, 4), dtype=float32,
# numpy array of x, y, w, h,; 2 - class labels tf.Tensor: shape=(3,), numpy array dtype=int32)>

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
