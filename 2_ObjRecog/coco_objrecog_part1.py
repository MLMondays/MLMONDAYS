
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


#----------------------------------------------
def get_inference_model(threshold, model):
    """
    get_inference_model(threshold, model)
    This function creates an inference model consisting of an input layer for an image
    the model predictions, decoded detections, then finally a mapping from image to detections
    In effect it is a model nested in another model
    INPUTS:
        * threshold [float], the detecton probability beyond which we are confident of
        * model [keras model], trained object detection model
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay
    OUTPUTS:  keras model for detections on images
    """
    # ANY size input
    image = tf.keras.Input(shape=[None, None, 3], name="image")

    predictions = model(image, training=False)

    detections = DecodePredictions(confidence_threshold=threshold)(image, predictions)

    inference_model = tf.keras.Model(inputs=image, outputs=detections)
    return inference_model


def lrfn(epoch):
    """
    lrfn(epoch)
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




###############################################################
## VARIABLES
###############################################################

sample_data_path = 'data/secoora/sample'
train_hist_fig = os.getcwd()+os.sep+'results/secoora_retinanet_model1.png'

model_dir = "retinanet/"
weights_dir = "data/coco"

train_csv = 'data/secoora/train_labels.csv'
sample_csv = 'data/secoora/sample.csv'


## data viz

sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))
print(len(sample_filenames))


dat = pd.read_csv(sample_csv)
print(len(dat))

grouped = split(dat, 'filename')

SAMPLE_NUM_PEOPLE = []
counter = 0
for group in grouped:
    image = file2tensor('data/secoora/sample/'+group[0])

    fig =plt.figure(figsize=(16,16))
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()

    bboxs = []
    labels = []
    for index, row in group.object.iterrows():
        labels.append(class_text_to_int(row['class']))
        bboxs.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        for box in bboxs:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            patch = plt.Rectangle([x1, y1], w, h, fill=False, edgecolor=[0, 1, 0], linewidth=1)
            ax.add_patch(patch)
            ax.text(x1, y1, 'person', bbox={"facecolor": [0, 1, 0], "alpha": 0.4}, clip_box=ax.clipbox, clip_on=True)
    #plt.show()
    plt.savefig('examples/secoora_examples'+str(counter)+'.png', dpi=200, bbox_inches='tight')
    counter += 1
    SAMPLE_NUM_PEOPLE.append(len(bboxs))



###############################################################
## EXECUTION
###############################################################

## custom learning rate scheduler

lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

rng = [i for i in range(MAX_EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, [lrfn(x) for x in rng])
# plt.show()
plt.savefig(os.getcwd()+os.sep+'results/learnratesched.png', dpi=200, bbox_inches='tight')



# ## Initializing model
# ## Building the ResNet50 backbone
# RetinaNet uses a ResNet based backbone, using which a feature pyramid network
# is constructed. In the example we use ResNet50 as the backbone, and return the
# feature maps at strides 8, 16 and 32.


resnet50_backbone = get_backbone()


# ## Implementing Smooth L1 loss and Focal Loss as keras custom losses

loss_fn = RetinaNetLoss(num_classes)

# ## Building the classification and box regression heads.
# The RetinaNet model has separate heads for bounding box regression and
# for predicting class probabilities for the objects. These heads are shared
# between all the feature maps of the feature pyramid.

model = RetinaNet(num_classes, resnet50_backbone)

"""
## Setting up callbacks, and compiling model
"""

earlystop = EarlyStopping(monitor="val_loss",
                              mode="min", patience=patience)


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

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

"""
## Building inference model
"""

threshold = 0.33 #0.5

inference_model = get_inference_model(threshold, model)


"""
## Generating detections
"""

plt.close('all')

## Load the COCO2017 dataset using TensorFlow Datasets

val_dataset, dataset_info = tfds.load("coco/2017", split="validation", data_dir="data",
                        download=False, with_info=True)

int2str = dataset_info.features["objects"]["label"].int2str




counter = 0
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

    visualize_detections(image, boxes, class_names, scores, counter, 'examples/coco_obrecog_part1_examples')
    counter +=1


## what about our secoora imagery?
sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))

print(len(sample_filenames))


# whats sorts of things might be we interested in?
SCORES =[] #probability of detection
NUM_PEOPLE = [] #number of people

for counter,f in enumerate(sample_filenames):
    image = file2tensor(f)

    image = tf.cast(image, dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]

    boxes = detections.nmsed_boxes[0][:num_detections] / ratio
    scores = detections.nmsed_scores[0][:num_detections]

    classes = ['person' for k in boxes]
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]

    visualize_detections(image, boxes, classes,scores, counter, 'examples/secoora_cocoweights_obrecog_part1_examples', figsize=(16, 16), linewidth=1, color=[0, 1, 0])
    SCORES.append(scores)
    NUM_PEOPLE.append(len(scores))

SCORES = np.hstack(SCORES)

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


# ## Setting up a `tf.data` pipeline
# To ensure that the model is fed with data efficiently we will be using
# `tf.data` API to create our input pipeline. The input pipeline
# consists for the following major processing steps:
# - Apply the preprocessing function to the samples
# - Create batches with fixed batch size. Since images in the batch can
# have different dimensions, and can also have different number of
# objects, we use `padded_batch` to the add the necessary padding to create
# rectangular tensors
# - Create targets for each sample in the batch using `LabelEncoderCoco`



# ## Preprocessing data
# Preprocessing the images involves two steps:
# - Resizing the image: Images are resized such that the shortest size is equal
# to 800 px, after resizing if the longest side of the image exceeds 1333 px,
# the image is resized such that the longest size is now capped at 1333 px.
# - Applying augmentation: Random scale jittering  and random horizontal flipping
# are the only augmentations applied to the images.
# Along with the images, bounding boxes are rescaled and flipped if required.
#



train_dataset, val_dataset = prepare_coco_datasets_for_training(train_dataset, val_dataset)


# epochs = 10

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

latest_checkpoint = tf.train.latest_checkpoint(model_dir)
model.load_weights(latest_checkpoint)

"""
## Building inference model amd test on secoora imagery again
"""

threshold = 0.33 #0.5

inference_model = get_inference_model(threshold, model)

SCORES2 =[] #probability of detection
NUM_PEOPLE2 = [] #number of people

for counter,f in enumerate(sample_filenames):
    image = file2tensor(f)

    image = tf.cast(image, dtype=tf.float32)
    input_image, ratio = prepare_image(image)
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]

    boxes = detections.nmsed_boxes[0][:num_detections] / ratio
    scores = detections.nmsed_scores[0][:num_detections]

    classes = ['person' for k in boxes]

    # visualize_detections(image, boxes, classes,scores)
    visualize_detections(image, boxes, classes,scores, counter, 'examples/secoora_finetunedweights_obrecog_part1_examples', figsize=(16, 16), linewidth=1, color=[0, 1, 0])
    SCORES2.append(scores)
    NUM_PEOPLE2.append(len(scores))

SCORES2 = np.hstack(SCORES2)


#any difference? plot number of detections and probability of thoem (histogram?)
