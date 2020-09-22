
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
# tf.keras.utils.get_file(filename, url)
#
#
# with zipfile.ZipFile("data.zip", "r") as z_fp:
#     z_fp.extractall("./")

#----------------------------------------------
def prepare_coco_datasets_for_training(train_dataset, val_dataset):

    train_dataset = train_dataset.map(preprocess_coco_data, num_parallel_calls=AUTO)

    train_dataset = train_dataset.shuffle(8 * BATCH_SIZE)
    train_dataset = train_dataset.padded_batch(
        batch_size = BATCH_SIZE, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )

    # for a in train_dataset.take(1):
    #     print(a)

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

#----------------------------------------------
def get_inference_model(threshold):

    # ANY size input
    image = tf.keras.Input(shape=[None, None, 3], name="image")

    predictions = model(image, training=False)

    detections = DecodePredictions(confidence_threshold=threshold)(image, predictions)

    inference_model = tf.keras.Model(inputs=image, outputs=detections)
    return inference_model



###############################################################
## VARIABLES
###############################################################

# data_path= os.getcwd()+os.sep+"data/tamucc/subset_3class/400"

sample_data_path = 'data/secoora/sample'

# filepath = os.getcwd()+os.sep+'results/tamucc_subset_3class_mv2_best_weights_model2.h5'

train_hist_fig = os.getcwd()+os.sep+'results/secoora_retinanet_model1.png'
# cm_filename = os.getcwd()+os.sep+'results/tamucc_sample_3class_mv2_model2_cm_val.png'
# sample_plot_name = os.getcwd()+os.sep+'results/tamucc_sample_3class_mv2_model2_est24samples.png'
#
# test_samples_fig = os.getcwd()+os.sep+'results/tamucc_full_sample_3class_mv2_model_est24samples.png'


model_dir = "retinanet/"
label_encoder = LabelEncoderCoco()

num_classes = 80

# learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
# learning_rate_boundaries = [125, 250, 500, 240000, 360000]
# learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
#     boundaries=learning_rate_boundaries, values=learning_rates
# )

start_lr = 1e-6 #0.00001
min_lr = start_lr
max_lr = 1e-3
rampup_epochs = 5
sustain_epochs = 0
exp_decay = .9
MAX_EPOCHS = 100
patience = 10


###############################################################
## EXECUTION
###############################################################

lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

rng = [i for i in range(MAX_EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, [lrfn(x) for x in rng])
# plt.show()
plt.savefig(os.getcwd()+os.sep+'results/learnratesched.png', dpi=200, bbox_inches='tight')



"""
## Initializing model
"""

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)


"""
## Setting up callbacks, and compiling model
"""

earlystop = EarlyStopping(monitor="val_loss",
                              mode="min", patience=patience)

# set checkpoint file
# model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
#                                 verbose=0, save_best_only=True, mode='min',
#                                 save_weights_only = True)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

callbacks = [model_checkpoint, earlystop, lr_callback]

optimizer = tf.optimizers.SGD(momentum=0.9)
# optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

weights_dir = "data/coco"
latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)



"""
## Building inference model
"""

threshold = 0.33 #0.5

inference_model = get_inference_model(threshold)

"""
## Generating detections
"""

plt.close('all')

## Load the COCO2017 dataset using TensorFlow Datasets

val_dataset, dataset_info = tfds.load("coco/2017", split="validation", data_dir="data",
                        with_info=True)

int2str = dataset_info.features["objects"]["label"].int2str

for sample in val_dataset.take(4):
    image = tf.cast(sample["image"], dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    boxes = detections.nmsed_boxes[0][:num_detections] / ratio
    scores = detections.nmsed_scores[0][:num_detections]

    visualize_detections(image, boxes, class_names, scores)




## what about our secoora imagery?
sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))

print(len(sample_filenames))

for counter,f in enumerate(sample_filenames):
    image = file2tensor(f)

    image = tf.cast(image, dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]

    boxes = detections.nmsed_boxes[0][:num_detections] / ratio
    scores = detections.nmsed_scores[0][:num_detections]

    classes = ['person' for k in boxes]

    visualize_detections(image, boxes, classes,scores)


# people not detected:
# in water
# on pier
# lying/sitting down
# groups
# with low probability


"""
## Training the model
"""

## Load the COCO2017 dataset using TensorFlow Datasets

#  set `data_dir=None` to load the complete dataset (huge download from internet)

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


train_dataset, val_dataset = prepare_coco_datasets_for_training(train_dataset, val_dataset)

epochs = 10

history = model.fit(
    train_dataset.take(50),
    validation_data=val_dataset.take(50),
    epochs=MAX_EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

# history.history.keys()

# Plot training history
plot_history(history, train_hist_fig)

plt.close('all')
K.clear_session()



# """
# ## Loading weights
# """
#
# # Change this to `model_dir` when not using the downloaded weights
# # weights_dir = "data/coco"
# #
# # # weights_dir = model_dir
# #
# # latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
# # model.load_weights(latest_checkpoint)

"""
## Building inference model amd test on secoora imagery again
"""

threshold = 0.33 #0.5

inference_model = get_inference_model(threshold)


for counter,f in enumerate(sample_filenames):
    image = file2tensor(f)

    image = tf.cast(image, dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]

    boxes = detections.nmsed_boxes[0][:num_detections] / ratio
    scores = detections.nmsed_scores[0][:num_detections]

    classes = ['person' for k in boxes]

    visualize_detections(image, boxes, classes,scores)


#any difference? plot number of detections and probability of thoem (histogram?)


# plt.close('all')
# val_dataset = tfds.load("coco/2017", split="validation", data_dir="data")
# int2str = dataset_info.features["objects"]["label"].int2str
#
# for sample in val_dataset.take(4):
#     image = tf.cast(sample["image"], dtype=tf.float32)
#     input_image, ratio = prepare_image(image)
#     detections = inference_model.predict(input_image)
#     num_detections = detections.valid_detections[0]
#     class_names = [
#         int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
#     ]
#     boxes = detections.nmsed_boxes[0][:num_detections] / ratio
#     scores = detections.nmsed_scores[0][:num_detections]
#
#     visualize_detections(image, boxes, class_names, scores)
#

    # visualize_detections(
    #     image,
    #     detections.nmsed_boxes[0][:num_detections] / ratio,
    #     class_names,
    #     detections.nmsed_scores[0][:num_detections],
    # )


#
# boxes = detections.nmsed_boxes[0][:num_detections] / ratio
# scores = detections.nmsed_scores[0][:num_detections]
# classes = ['','','']
# image = np.array(image, dtype=np.uint8)
# linewidth=1
# color=[0, 0, 1]
#
#
# plt.figure(figsize=(7, 7))
# plt.axis("off")
# plt.imshow(image)
# ax = plt.gca()
# for box, cls, score in zip(boxes.numpy(), classes, scores):
#     print(box)
#
#     text = "{}: {:.2f}".format(cls, score)
#     x1, y1, x2, y2 = box
#     w, h = x2 - x1, y2 - y1
#     patch = plt.Rectangle(
#         [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
#     )
#     ax.add_patch(patch)
#     ax.text(
#         x1,
#         y1,
#         text,
#         bbox={"facecolor": color, "alpha": 0.4},
#         clip_box=ax.clipbox,
#         clip_on=True,
#     )
# plt.show()



# evaluation
# iou
# viz - examples of classes (small tiles)


    #
    # image = np.array(image, dtype=np.uint8)
    # plt.figure(figsize=(7, 7))
    # plt.axis("off")
    # plt.imshow(image)
    # ax = plt.gca()
    # for box, _cls, score in zip(boxes, classes, scores):
    #     text = "{}: {:.2f}".format(_cls, score)
    #     x1, y1, x2, y2 = box
    #     w, h = x2 - x1, y2 - y1
    #     patch = plt.Rectangle(
    #         [x1, y1], w, h, fill=False, edgecolor=[1,0,0], linewidth=2
    #     )
    #     ax.add_patch(patch)
    #     ax.text(
    #         x1,
    #         y1,
    #         text,
    #         bbox={"facecolor": [0,1,0], "alpha": 0.4},
    #         clip_box=ax.clipbox,
    #         clip_on=True,
    #     )
    # plt.show()
