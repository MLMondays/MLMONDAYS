
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

from imports import *

"""
## Setting up training parameters
"""

model_dir = "retinanet/"

num_classes = 80
BATCH_SIZE = 1

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

data_path = "/media/marda/TWOTB/USGS/SOFTWARE/DL-CDI2020/2_ObjRecog/data/secoora"

AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API


"""
## Initializing and compiling model
"""

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)


weights_dir = "data/coco"
latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

"""
## Setting up callbacks
"""

# earlystop = EarlyStopping(monitor="val_loss",
#                               mode="min", patience=patience)
#
# # set checkpoint file
# model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
#                                 verbose=0, save_best_only=True, mode='min',
#                                 save_weights_only = True)
#
# callbacks = [model_checkpoint, earlystop, lr_callback]
#
# do_train = False #True
#
# if do_train:

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
]

"""
## Load the Secoora dataset

"""


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

# for a in train_dataset.take(1):
#     print(a[0])


val_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*val*.tfrecord'))
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
#
# for a in val_dataset.take(1):
#     print(a[0])


"""
## Training the model
"""

#load coco weights and build on top

# run for a few epochs and a subset of the take to check the losses are going in the right direction
epochs = 10

model.fit(
    train_dataset.take(50),
    validation_data=val_dataset.take(50),
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)

## looks good. now we'll train on the full dataset, and for longer
epochs = 100

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)








#under-estimates number of people, and low confidence in predictions

# dataviz to show this?






# ## vizualize images and biounding boxes1

#
# ## resnet50_backbone = get_backbone()
# loss_fn = RetinaNetLoss(num_classes)
# model = RetinaNet(num_classes, resnet50_backbone)
#
# optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
# model.compile(loss=loss_fn, optimizer=optimizer)
#
#
# # weights_dir = "data/coco"
# #
# weights_dir = model_dir
# #
# # latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
# model.load_weights(latest_checkpoint)


"""
## Building inference model
"""

image = tf.keras.Input(shape=[None, None, 3], name="image")
# Out[13]: <tf.Tensor 'image_1:0' shape=(None, None, None, 3) dtype=float32>

predictions = model(image, training=False)
# Out[12]: <tf.Tensor 'RetinaNet_1/Identity:0' shape=(None, None, 84) dtype=float32>


threshold = 0.33

detections = DecodePredictions(confidence_threshold=threshold)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)

"""
## Generating detections
"""

# val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
# int2str = dataset_info.features["objects"]["label"].int2str

val_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*val*.tfrecord'))
val_dataset = tf.data.TFRecordDataset(val_filenames)
val_dataset = val_dataset.map(_parse_function)

for sample in val_dataset.take(2):
    # image = tf.squeeze(tf.cast(sample[0], dtype=tf.float32))

    image = tf.image.decode_jpeg(sample['image'], channels=3)
    image = tf.cast(image, tf.float32)

    input_image, ratio = prepare_image(image)
    #detections = inference_model.predict(tf.squeeze(input_image))
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]

    boxes = detections.nmsed_boxes[0][:num_detections] / ratio
    scores = detections.nmsed_scores[0][:num_detections]

    classes = ['person' for k in boxes]

    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=(7, 7))
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=[1,0,0], linewidth=2
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": [0,1,0], "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()


##read image from secoora sample directory

sample_data_path = 'data/secoora/sample'

sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))


plt.figure(figsize=(16,16))

for counter,f in enumerate(sample_filenames):
    image = file2tensor(f)

    image = tf.cast(image, dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]

    boxes = detections.nmsed_boxes[0][:num_detections] / ratio
    scores = detections.nmsed_scores[0][:num_detections]

    classes = ['person' for k in boxes]

    image = np.array(image, dtype=np.uint8)
    plt.figure(figsize=(7, 7))
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=[1,0,0], linewidth=2
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": [0,1,0], "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
