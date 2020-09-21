
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

# """
# ## Downloading the COCO2017 dataset
# Training on the entire COCO2017 dataset which has around 118k images takes a
# lot of time, hence we will be using a smaller subset of ~500 images for
# training in this example.
# """
#
# url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
# filename = os.path.join(os.getcwd(), "data.zip")
# keras.utils.get_file(filename, url)
#
#
# with zipfile.ZipFile("data.zip", "r") as z_fp:
#     z_fp.extractall("./")


"""
## Setting up training parameters
"""

model_dir = "retinanet/"
label_encoder = LabelEncoderCoco()

num_classes = 80
# batch_size = 1

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

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
## Load the COCO2017 dataset using TensorFlow Datasets
"""

#  set `data_dir=None` to load the complete dataset

(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
)

"""
## Setting up a `tf.data` pipeline
To ensure that the model is fed with data efficiently we will be using
`tf.data` API to create our input pipeline. The input pipeline
consists for the following major processing steps:
- Apply the preprocessing function to the samples
- Create batches with fixed batch size. Since images in the batch can
have different dimensions, and can also have different number of
objects, we use `padded_batch` to the add the necessary padding to create
rectangular tensors
- Create targets for each sample in the batch using `LabelEncoderCoco`
"""

# autotune = tf.data.experimental.AUTOTUNE
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

# for a, b in train_dataset.take(1):
#     print(a)

val_dataset = val_dataset.map(preprocess_coco_data, num_parallel_calls=AUTO)
val_dataset = val_dataset.padded_batch(
    batch_size = BATCH_SIZE, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=AUTO)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(AUTO)

"""
## Training the model
"""

# Uncomment the following lines, when training on full dataset
# train_steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
# val_steps_per_epoch = \
#     dataset_info.splits["validation"].num_examples // BATCH_SIZE

# train_steps = 4 * 100000
# epochs = train_steps // train_steps_per_epoch

# epochs = 1

# Running 100 training and 50 validation steps,
# remove `.take` when training on the full dataset

# model.fit(
#     train_dataset.take(50),
#     validation_data=val_dataset.take(50),
#     epochs=MAX_EPOCHS,
#     callbacks=callbacks_list,
#     verbose=1,
# )

"""
## Loading weights
"""

# Change this to `model_dir` when not using the downloaded weights
weights_dir = "data/coco"

# weights_dir = model_dir

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

"""
## Building inference model
"""

image = tf.keras.Input(shape=[None, None, 3], name="image")
# Out[8]: <tf.Tensor 'image:0' shape=(None, None, None, 3) dtype=float32>


predictions = model(image, training=False)
# Out[4]: <tf.Tensor 'RetinaNet/Identity:0' shape=(None, None, 84) dtype=float32>


detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
# CombinedNonMaxSuppression(nmsed_boxes=<tf.Tensor 'decode_predictions/Identity:0' shape=(None, 100, 4) dtype=float32>,
# nmsed_scores=<tf.Tensor 'decode_predictions/Identity_1:0' shape=(None, 100) dtype=float32>,
# nmsed_classes=<tf.Tensor 'decode_predictions/Identity_2:0' shape=(None, 100) dtype=float32>,
# valid_detections=<tf.Tensor 'decode_predictions/Identity_3:0' shape=(None,) dtype=int32>)


inference_model = tf.keras.Model(inputs=image, outputs=detections)

"""
## Generating detections
"""

val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
int2str = dataset_info.features["objects"]["label"].int2str

for sample in val_dataset.take(2):
    image = tf.cast(sample["image"], dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )



boxes = detections.nmsed_boxes[0][:num_detections] / ratio
scores = detections.nmsed_scores[0][:num_detections]
classes = []
image = np.array(image, dtype=np.uint8)
linewidth=1
color=[0, 0, 1]


plt.figure(figsize=(7, 7))
plt.axis("off")
plt.imshow(image)
ax = plt.gca()
for box, _cls, score in zip(boxes.numpy(), classes, scores):
    text = "{}: {:.2f}".format(_cls, score)
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    patch = plt.Rectangle(
        [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
    )
    ax.add_patch(patch)
    ax.text(
        x1,
        y1,
        text,
        bbox={"facecolor": color, "alpha": 0.4},
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
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )

# evaluation
# iou
# viz - examples of classes (small tiles)


    #
    #     plt.subplot(6,4,counter+1)
    #     name = sample_filenames[counter].split(os.sep)[-1].split('_')[0]
    #     plt.title(name, fontsize=10)
    #     plt.imshow(tf.cast(image, tf.uint8))
    #     plt.axis('off')
    #
    #     scores = model.predict(tf.expand_dims(im, 0) , batch_size=1)
    #     n = np.argmax(scores[0])
    #     est_name = CLASSES[n].decode()
    #     if name==est_name:
    #        plt.text(10,50,'prediction: %s' % est_name,
    #                 color='k', fontsize=12,
    #                 ha="center", va="center",
    #                 bbox=dict(boxstyle="round",
    #                        ec=(.1, 1., .5),
    #                        fc=(.1, 1., .5),
    #                        ))
    #     else:
    #        plt.text(10,50,'prediction: %s' % est_name,
    #                 color='k', fontsize=12,
    #                 ha="center", va="center",
    #                 bbox=dict(boxstyle="round",
    #                        ec=(1., 0.5, 0.1),
    #                        fc=(1., 0.8, 0.8),
    #                        ))
    #
    # # plt.show()
    # plt.savefig(test_samples_fig,
    #             dpi=200, bbox_inches='tight')
    # plt.close('all')
