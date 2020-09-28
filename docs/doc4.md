---
id: doc4
title: ML Mondays API
---

Page under construction -- please check back later

This is the class and function reference of the ML Mondays course code

## 1_ImageRecog

### General workflow using your own data

1. Create a TFREcord dataset from your data, organised as follows:
  * copy training images into a folder called `train`
  * copy validation images into a folder called `validation`
  * ensure the class name is written to each file name. Ideally this is a prefix such that it is trivial to extract the class name from the file name
  * modify one of the provided workflows (such as `tamucc_make_tfrecords.py`) for your dataset, to create your train and validation tfrecord shards

2. Set up your model
  * Decide on whether you want to train a small custom model from scratch, a large model from scratch, or a large model trained using weights transfered from another task
  * If a small custom model, use `make_cat_model` with `shallow=True` for a relatively small model, and `shallow=False` for a relatively large model
  * If a large model with transfer learning, decide on which one to utilize (`transfer_learning_mobilenet_model`, `transfer_learning_xception_model`, or `transfer_learning_model_vgg`)
  * If you wish to train a large model from scratch, decide on which one to utilize (`mobilenet_model`, or `xception_model`)

3. Set up a data pipeline
  * Modify and follow the provided examples to create a `get_training_dataset()` and `get_validation_dataset()`. This will likely require you copy and modify `get_batched_dataset` to your own needs, depending on the format of your labels in filenames, by writing your own `read_tfrecord` function for your dataset (depending on the model selected)

4. Set up a model training pipeline
  * `.compile()` your model with an appropriate loss function and metrics
  * define a `LearningRateScheduler` function to vary learning rates over training as a function of training epoch
  * define an `EarlyStopping` criteria and create a `ModelCheckpoint` to save trained model weights
  * if transfer learning using weights not from imagenet, load your initial weights from somewhere else

5. Train the model
  * Use `history = model.fit()` to create a record of the training history. Pass the training and validation datasets, and a list of callbacks containing your model checkpoint, learning rate scheduler, and early stopping monitor)

6. Evaluate your model
  * Plot and study the `history` time-series of losses and metrics. If unsatisfactory, begin the iterative process of model optimization  
  * Use the `loss, accuracy = model.evaluate(get_validation_dataset(), batch_size=BATCH_SIZE, steps=validation_steps)` function using the validation dataset and specifying the number of validation steps
  * Make plots of model outputs, organized in such a way that you can at-a-glance see where the model is failing. Make use of `make_sample_plot` and `p_confmat`, as a starting point, to visualize sample imagery with their model predictions, and a confusion matrix of predicted/true class-correspondences


### model_funcs.py

#### Model creation

---
##### transfer_learning_model_vgg
```python
transfer_learning_model_vgg(num_classes, input_shape, dropout_rate=0.5)
```

This function creates an implementation of a convolutional deep learning model for estimating
a discrete category based on vgg, trained using transfer learning
(initialized using pretrained imagenet weights)

* INPUTS:
    * `num_classes` = number of classes (output nodes on classification layer)
    * `input_shape` = size of input layer (i.e. image tensor)
* OPTIONAL INPUTS:
    * `dropout_rate` = proportion of neurons to randomly set to zero, after the pooling layer
* GLOBAL INPUTS: None
* OUTPUTS: keras model instance

---
##### mobilenet_model
```python
mobilenet_model(num_classes, input_shape, dropout_rate=0.5)
```

This function creates an implementation of a convolutional deep learning model for estimating
a discrete category based on mobilenet, trained from scratch

* INPUTS:
    * `num_classes` = number of classes (output nodes on classification layer)
    * `input_shape` = size of input layer (i.e. image tensor)
* OPTIONAL INPUTS:
    * `dropout_rate` = proportion of neurons to randomly set to zero, after the pooling layer
* GLOBAL INPUTS: None
* OUTPUTS: keras model instance

---
##### transfer_learning_mobilenet_model
```python
transfer_learning_mobilenet_model(num_classes, input_shape, dropout_rate=0.5)
```

This function creates an implementation of a convolutional deep learning model for estimating
a discrete category based on mobilenet v2, trained using transfer learning
(initialized using pretrained imagenet weights)

* INPUTS:
    * `num_classes` = number of classes (output nodes on classification layer)
    * `input_shape` = size of input layer (i.e. image tensor)
* OPTIONAL INPUTS:
    * `dropout_rate` = proportion of neurons to randomly set to zero, after the pooling layer
* GLOBAL INPUTS: None
* OUTPUTS: keras model instance


---
##### transfer_learning_xception_model
```python
transfer_learning_xception_model(num_classes, input_shape, dropout_rate=0.25)
```

This function creates an implementation of a convolutional deep learning model for estimating
a discrete category based on xception, trained using transfer learning
(initialized using pretrained imagenet weights)

* INPUTS:
    * `num_classes` = number of classes (output nodes on classification layer)
    * `input_shape` = size of input layer (i.e. image tensor)
* OPTIONAL INPUTS:
    * `dropout_rate` = proportion of neurons to randomly set to zero, after the pooling layer
* GLOBAL INPUTS: None
* OUTPUTS: keras model instance


---
##### xception_model
```python
xception_model(num_classes, input_shape, dropout_rate=0.25)
```

This function creates an implementation of a convolutional deep learning model for estimating
a discrete category based on xception, trained from scratch

* INPUTS:
    * `num_classes` = number of classes (output nodes on classification layer)
    * `input_shape` = size of input layer (i.e. image tensor)
* OPTIONAL INPUTS:
    * `dropout_rate` = proportion of neurons to randomly set to zero, after the pooling layer
* GLOBAL INPUTS: None
* OUTPUTS: keras model instance

---
##### conv_block
```python
conv_block(inp, filters=32, bn=True, pool=True)
```

This function generates a convolutional block

* INPUTS:
    * `inp` = input layer
* OPTIONAL INPUTS:
    * `filters` = number of convolutional filters to use
    * `bn`=False, use batch normalization in each convolutional layer
    * `pool`=True, use pooling in each convolutional layer
    * `shallow`=True, if False, a larger model with more convolution layers is used
* GLOBAL INPUTS: None
* OUTPUTS: keras model layer object

---
##### make_cat_model
```python
make_cat_model(num_classes, dropout, denseunits, base_filters, bn=False, pool=True, shallow=True)
```

This function creates an implementation of a convolutional deep learning model for estimating
a discrete category

* INPUTS:
    * `num_classes` = number of classes (output nodes on classification layer)
    * `dropout` = proportion of neurons to randomly set to zero, after the pooling layer
    * `denseunits` = number of neurons in the classifying layer
    * `base_filters` = number of convolutional filters to use in the first layer
* OPTIONAL INPUTS:
    * `bn`=False, use batch normalization in each convolutional layer
    * `pool`=True, use pooling in each convolutional layer
    * `shallow`=True, if False, a larger model with more convolution layers is used
* GLOBAL INPUTS: TARGET_SIZE
* OUTPUTS: keras model instance


#### Model training

---
##### lrfn
```python
lrfn(epoch)
```

This function creates a custom piecewise linear-exponential learning rate function for a custom learning rate scheduler. It is linear to a max, then exponentially decays

* INPUTS: current `epoch` number
* OPTIONAL INPUTS: None
* GLOBAL INPUTS:`start_lr`, `min_lr`, `max_lr`, `rampup_epochs`, `sustain_epochs`, `exp_decay`
* OUTPUTS:  the function lr with all arguments passed


### tfrecords_funcs.py

#### TF-dataset creation

---
##### get_batched_dataset
```python
get_batched_dataset(filenames)
```

This function defines a workflow for the model to read data from
tfrecord files by defining the degree of parallelism, batch size, pre-fetching, etc
and also formats the imagery properly for model training
(assumes mobilenet by using read_tfrecord_mv2)

* INPUTS:
    * `filenames` [list]
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: BATCH_SIZE, AUTO
* OUTPUTS: `tf.data.Dataset` object

---
##### get_eval_dataset
```python
get_eval_dataset(filenames)
```

This function defines a workflow for the model to read data from
tfrecord files by defining the degree of parallelism, batch size, pre-fetching, etc
and also formats the imagery properly for model training
(assumes mobilenet by using read_tfrecord_mv2). This evaluation version does not .repeat() because it is not being called repeatedly by a model

* INPUTS:
    * `filenames` [list]
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: BATCH_SIZE, AUTO
* OUTPUTS: `tf.data.Dataset` object

---
##### resize_and_crop_image
```python
resize_and_crop_image(image, label)
```

This function crops to square and resizes an image. The label passes through unmodified

* INPUTS:
    * `image` [tensor array]
    * `label` [int]
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: TARGET_SIZE
* OUTPUTS:
    * `image` [tensor array]
    * `label` [int]

---
##### recompress_image
```python
recompress_image(image, label)
```

This function takes an image encoded as a byte string and recodes as an 8-bit jpeg. Label passes through unmodified

* INPUTS:
    * `image` [tensor array]
    * `label` [int]
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * `image` [tensor array]
    * `label` [int]


#### TFRecord reading

---
##### file2tensor
```python
file2tensor(f, model='mobilenet')
```

This function reads a jpeg image from file into a cropped and resized tensor,
for use in prediction with a trained mobilenet or vgg model
(the imagery is standardized depending on target model framework)

* INPUTS:
    * `f` [string] file name of jpeg
* OPTIONAL INPUTS:
    * `model` = {'mobilenet' | 'vgg'}
* OUTPUTS:
    * `image` [tensor array]: unstandardized image
    * `im` [tensor array]: standardized image
* GLOBAL INPUTS: TARGET_SIZE

---
##### read_classes_from_json
```python
read_classes_from_json(json_file)
```

This function reads the contents of a json file enumerating classes

* INPUTS:
    * `json_file` [string]: full path to the json file
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: `CLASSES` [list]: list of classesd as byte strings

---
##### read_tfrecord_vgg
```python
read_tfrecord_vgg(example)
```

This function reads an example record from a tfrecord file
and parses into label and image ready for vgg model training

* INPUTS:
    * `example`: an tfrecord 'example' object, containing an image and label
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: TARGET_SIZE
* OUTPUTS:
    * `image` [tensor]: resized and pre-processed for vgg
    * `class_label` [tensor] 32-bit integer


---
##### read_tfrecord_mv2
```python
read_tfrecord_mv2(example)
```

This function reads an example record from a tfrecord file
and parses into label and image ready for mobilenet model training

* INPUTS:
    * `example`: an tfrecord 'example' object, containing an image and label
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: TARGET_SIZE
* OUTPUTS:
    * `image` [tensor]: resized and pre-processed for mobilenetv2
    * `class_label` [tensor] 32-bit integer


---
##### read_tfrecord
```python
read_tfrecord(example)
```

This function reads an example from a TFrecord file into a single image and label

* INPUTS:
    * TFRecord `example` object
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: TARGET_SIZE
* OUTPUTS:
    * `image` [tensor array]
    * `class_label` [tensor int]

---
##### read_image_and_label
```python
read_image_and_label(img_path)
```

This function reads a jpeg image from a provided filepath and extracts the label from the filename (assuming the class name is before "IMG" in the filename)

* INPUTS:
    * `img_path` [string]: filepath to a jpeg image
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * `image` [tensor array]
    * `class_label` [tensor int]


#### TFRecord creation

---
##### get_dataset_for_tfrecords
```python
get_dataset_for_tfrecords(recoded_dir, shared_size)
```

This function reads a list of TFREcord shard files, decode the images and label resize and crop the image to TARGET_SIZE, and create batches

* INPUTS:
    * `recoded_dir`
    * `shared_size`
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: TARGET_SIZE
* OUTPUTS: `tf.data.Dataset` object

---
##### write_records
```python
write_records(tamucc_dataset, tfrecord_dir, CLASSES)
```

This function writes a tf.data.Dataset object to TFRecord shards

* INPUTS:
    * `tamucc_dataset` [tf.data.Dataset]
    * `tfrecord_dir` [string] : path to directory where files will be written
    * `CLASSES` [list] of class string names
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: None (files written to disk)

---
##### to_tfrecord
```python
to_tfrecord(img_bytes, label, CLASSES)
```

This function creates a TFRecord example from an image byte string and a label feature

* INPUTS:
    * `img_bytes`: an image bytestring
    * `label`: label string of image
    * `CLASSES`: list of string classes in the entire dataset
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: tf.train.Feature example


### plot_funcs.py

---
##### plot_history
```python
plot_history(history, train_hist_fig)
```

This function plots the training history of a model

* INPUTS:
    * `history` [dict]: the output dictionary of the model.fit() process, i.e. history = model.fit(...)
    * `train_hist_fig` [string]: the filename where the plot will be printed
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: None (figure printed to file)

---
##### get_label_pairs
```python
get_label_pairs(val_ds, model)
```

This function gets label observations and model estimates

* INPUTS:
    * `val_ds`: a batched data set object
    * `model`: trained and compiled keras model instance
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * `labs` [ndarray]: 1d vector of numeric labels
    * `preds` [ndarray]: 1d vector of correspodning model predicted numeric labels

---
##### p_confmat
```python
p_confmat(labs, preds, cm_filename, CLASSES, thres = 0.1)
```

This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
saving out to the provided filename, cm_filename

* INPUTS:
    * `labs` [ndarray]: 1d vector of labels
    * `preds` [ndarray]: 1d vector of model predicted labels
    * `cm_filename` [string]: filename to write the figure to
    * `CLASSES` [list] of strings: class names
* OPTIONAL INPUTS:
    * `thres` [float]: threshold controlling what values are displayed
* GLOBAL INPUTS: None
* OUTPUTS: None (figure printed to file)

---
##### make_sample_plot
```python
make_sample_plot(model, sample_filenames, test_samples_fig, CLASSES))
```

This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
saving out to the provided filename, cm_filename

* INPUTS:
    * `model`: trained and compiled keras model
    * `sample_filenames`: [list] of strings
    * `test_samples_fig` [string]: filename to print figure to
    * `CLASSES` [list] os trings: class names
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: None (matplotlib figure, printed to file)

---
##### compute_hist
```python
compute_hist(images)
```

Compute the per channel histogram for a batch
of images

* INPUTS:
    * `images` [ndarray]: batch of shape (N x W x H x 3)
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * `hist_r` [dict]: histogram frequencies {'hist'} and bins {'bins'} for red channel
    * `hist_g` [dict]: histogram frequencies {'hist'} and bins {'bins'} for green channel
    * `hist_b` [dict]: histogram frequencies {'hist'} and bins {'bins'} for blue channel

---
##### plot_distribution
```python
plot_distribution(images, labels, class_id, CLASSES)
```

Compute the per channel histogram for a batch of images

* INPUTS:
    * `images` [ndarray]: batch of shape (N x W x H x 3)
    * `labels` [ndarray]: batch of shape (N x 1)
    * `class_id` [int]: class integer to plot
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: matplotlib figure

---
##### plot_one_class
```python
plot_one_class(inp_batch, sample_idx, label, batch_size, CLASSES, rows=8, cols=8, size=(20,15))
```

Plot `batch_size` images that belong to the class `label`

* INPUTS:
    * `inp_batch` [ndarray]: batch of N images
    * `sample_idx` [list]: indices of the N images
    * `label` [string]: class string
    * `batch_size` [int]: number of images to plot
* OPTIONAL INPUTS:
    * `rows`=8 [int]: number of rows
    * `cols`=8 [int]: number of columns
    * `size`=(20,15) [tuple]: size of matplotlib figure
* GLOBAL INPUTS: None (matplotlib figure, printed to file)

---
##### compute_mean_image
```python
compute_mean_image(images, opt="mean")
```

Compute and return mean image given a batch of images

* INPUTS:
    * `images` [ndarray]: batch of shape (N x W x H x 3)
* OPTIONAL INPUTS:
    * `opt`="mean" or "median"
* GLOBAL INPUTS:
* OUTPUTS: 2d mean image [ndarray]

---
##### plot_mean_images
```python
plot_mean_images(images, labels, CLASSES, rows=3, cols = 2)
```

Plot the mean image of a set of images

* INPUTS:
    * `images` [ndarray]: batch of shape (N x W x H x 3)
    * `labels` [ndarray]: batch of shape (N x 1)
* OPTIONAL INPUTS:
  * `rows` [int]: number of rows
  * `cols` [int]: number of columns
* GLOBAL INPUTS: CLASSES
* OUTPUTS: matplotlib figure

---
##### plot_tsne
```python
plot_tsne(tsne_result, label_ids, CLASSES)
```

Plot TSNE loadings and colour code by class. [Source](https://www.kaggle.com/gaborvecsei/plants-t-sne)

* INPUTS:
  * `tsne_result` [ndarray]: N x 2 data of loadings on two axes
  * `label_ids` [int]: N class labels
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: CLASSES
* OUTPUTS: matplotlib figure, matplotlib figure axes object

---
##### visualize_scatter_with_images
```python
visualize_scatter_with_images(X_2d_data, labels, images, figsize=(15,15), image_zoom=1,xlim = (-3,3), ylim=(-3,3))
```

Plot TSNE loadings and colour code by class. [Source](https://www.kaggle.com/gaborvecsei/plants-t-sne)

* INPUTS:
    * `X_2d_data` [ndarray]: N x 2 data of loadings on two axes
    * `images` [ndarray] : N batch of images to plot
* OPTIONAL INPUTS:
    * `figsize`=(15,15)
    * `image_zoom`=1 [float]: control the scaling of the imagery (make smaller for smaller thumbnails)
    * `xlim` = (-3,3) [tuple]: set x axes limits
    * `ylim` = (-3,3) [tuple]: set y axes limits]
* GLOBAL INPUTS: None
* OUTPUTS: matplotlib figure



## 2_ObjRecog

### General workflow using your own data

1. Create a TFREcord dataset from your data, organised as follows:
  * copy training images into a folder called `train`
  * copy validation images into a folder called `validation`
  * create a text, csv file that lists each of the objects in each image, with the following columns: filename, xmin, ymin, xmax (float y coord pixel), ymax (float y coord pixel), class (string)
  * modify the provided workflow (`secoora_make_tfrecords.py`) for your dataset, to create your train and validation tfrecord shards

```
  filename,	width,	height,	class,	xmin,	ymin,	xmax,	ymax
  staugustinecam.2019-04-18_1400.mp4_frames_25.jpg,	1280,	720,	person,	1088,	581,	1129,	631
  staugustinecam.2019-04-18_1400.mp4_frames_25.jpg,	1280,	720,	person,	1125,	524,	1183,	573
  staugustinecam.2019-04-04_0700.mp4_frames_51.jpg,	1280,	720,	person,	158,	198,	178,	244
  staugustinecam.2019-04-04_0700.mp4_frames_51.jpg,	1280,	720,	person,	131,	197,	162,	244
  staugustinecam.2019-04-04_0700.mp4_frames_51.jpg,	1280,	720,	person,	40,	504,	87,	581
  staugustinecam.2019-04-04_0700.mp4_frames_51.jpg,	1280,	720,	person,	0,	492,	15,	572
  staugustinecam.2019-01-01_1400.mp4_frames_44.jpg,	1280,	720,	person,	1086,	537,	1130,	615
  staugustinecam.2019-01-01_1400.mp4_frames_44.jpg,	1280,	720,	person,	1064,	581,	1134,	624
  staugustinecam.2019-01-01_1400.mp4_frames_44.jpg,	1280,	720,	person,	1136,	526,	1186,	570
```

2. Set up your model
  * Decide on whether you want to train a model from scratch, or trained using weights transfered from another task (such as coco 2017)

3. Set up a model training pipeline
  * `.compile()` your model with an appropriate loss function and metrics
  * define a `LearningRateScheduler` function to vary learning rates over training as a function of training epoch
  * define an `EarlyStopping` criteria and create a `ModelCheckpoint` to save trained model weights
  * if transfer learning using weights not from coco, load your initial weights from somewhere else

5. Train the model
  * Use `history = model.fit()` to create a record of the training history. Pass the training and validation datasets, and a list of callbacks containing your model checkpoint, learning rate scheduler, and early stopping monitor)

6. Evaluate your model
  * Plot and study the `history` time-series of losses and metrics. If unsatisfactory, begin the iterative process of model optimization  
  * Use the `loss, accuracy = model.evaluate(get_validation_dataset(), batch_size=BATCH_SIZE, steps=validation_steps)` function using the validation dataset and specifying the number of validation steps
  * Make plots of model outputs, organized in such a way that you can at-a-glance see where the model is failing. Make use of `visualize_detections`, as a starting point, to visualize sample imagery with their model predictions


### model_funcs.py

#### Model creation

---
##### AnchorBox
```python
AnchorBox()
```

Code from https://keras.io/examples/vision/retinanet/. Generates anchor boxes.
This class has operations to generate anchor boxes for feature maps at strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the format `[x, y, width, height]`.

* INPUTS:
  * `aspect_ratios`: A list of float values representing the aspect ratios of
    the anchor boxes at each location on the feature map
  * `scales`: A list of float values representing the scale of the anchor boxes
    at each location on the feature map.
  * `num_anchors`: The number of anchor boxes at each location on feature map
  * `areas`: A list of float values representing the areas of the anchor
    boxes for each feature map in the feature pyramid.
  * `strides`: A list of float value representing the strides for each feature
    map in the feature pyramid.
* OPTIONAL INPUTS: None
* OUTPUTS: anchor boxes for all the feature maps, stacked as a single tensor with shape
    `(total_anchors, 4)`, when `AnchorBox._get_anchors()` is called
* GLOBAL INPUTS: None

---
##### get_backbone
```python
get_backbone()
```

Code from https://keras.io/examples/vision/retinanet/. This function Builds ResNet50 with pre-trained imagenet weights

* INPUTS: None
* OPTIONAL INPUTS: None
* OUTPUTS: keras Model
* GLOBAL INPUTS: BATCH_SIZE

---
##### FeaturePyramid
```python
FeaturePyramid()
```

Code from https://keras.io/examples/vision/retinanet/. This class builds the Feature Pyramid with the feature maps from the backbone.

* INPUTS:
  * `num_classes`: Number of classes in the dataset.
  * `backbone`: The backbone to build the feature pyramid from. Currently supports ResNet50 only (the output of get_backbone())
* OPTIONAL INPUTS: None
* OUTPUTS: the 5-feature pyramids (feature maps) at strides `[8, 16, 32, 64, 128]`
* GLOBAL INPUTS: None

---
##### build_head
```python
build_head(output_filters, bias_init)
```

Code from https://keras.io/examples/vision/retinanet/. This function builds the class/box predictions head.

* INPUTS:
    * `output_filters`: Number of convolution filters in the final layer.
    * `bias_init`: Bias Initializer for the final convolution layer.
* OPTIONAL INPUTS: None
* OUTPUTS: a keras sequential model representing either the classification
      or the box regression head depending on `output_filters`.
* GLOBAL INPUTS: None

---
##### RetinaNet
```python
RetinaNet()
```

Code from https://keras.io/examples/vision/retinanet/. This class returns a subclassed Keras model implementing the RetinaNet architecture.

* INPUTS:
    * `num_classes`: Number of classes in the dataset.
    * `backbone`: The backbone to build the feature pyramid from. Supports ResNet50 only.
* OPTIONAL INPUTS: None
* OUTPUTS:
    * `val_dataset` [tensorflow dataset]: validation dataset
    * `train_dataset` [tensorflow dataset]: training dataset
* GLOBAL INPUTS: None

#### Model training

---
##### compute_iou
```python
compute_iou(boxes1, boxes2)
```

This function computes pairwise IOU matrix for given two sets of boxes

* INPUTS:
    * `boxes1`: A tensor with shape `(N, 4)` representing bounding boxes
      where each box is of the format `[x, y, width, height]`.
    * `boxes2`: A tensor with shape `(M, 4)` representing bounding boxes
      where each box is of the format `[x, y, width, height]`.
* OPTIONAL INPUTS: None
* OUTPUTS: pairwise IOU matrix with shape `(N, M)`, where the value at ith row jth column holds the IOU between ith box and jth box from `boxes1` and `boxes2` respectively.
* GLOBAL INPUTS: None

---
##### RetinaNetBoxLoss
```python
RetinaNetBoxLoss()
```

Code from https://keras.io/examples/vision/retinanet/. This class implements smooth L1 loss

* INPUTS:
    * `y_true` [tensor]: label observations
    * `y_pred` [tensor]: label estimates
* OPTIONAL INPUTS: None
* OUTPUTS:
    * `loss` [tensor]
* GLOBAL INPUTS: None

---
##### RetinaNetClassificationLoss
```python
RetinaNetClassificationLoss()
```

Code from https://keras.io/examples/vision/retinanet/. This class implements Focal loss.

* INPUTS:
    * `y_true` [tensor]: label observations
    * `y_pred` [tensor]: label estimates
* OPTIONAL INPUTS: None
* OUTPUTS:
    * `loss` [tensor]
* GLOBAL INPUTS: None

---
##### RetinaLoss
```python
RetinaNetLoss()
```

Code from https://keras.io/examples/vision/retinanet/. This class is a wrapper to sum RetinaNetClassificationLoss and RetinaNetClassificationLoss outputs.

* INPUTS:
    * `y_true` [tensor]: label observations
    * `y_pred` [tensor]: label estimates
* OPTIONAL INPUTS: None
* OUTPUTS:
    * `loss` [tensor]
* GLOBAL INPUTS: None


#### Model prediction

---
##### DecodePredictions
```python
DecodePredictions()
```

Code from https://keras.io/examples/vision/retinanet/. This class creates a Keras layer that decodes predictions of the RetinaNet model.

* INPUTS:
    * `num_classes`: Number of classes in the dataset
    * `confidence_threshold`: Minimum class probability, below which detections
      are pruned.
    * `nms_iou_threshold`: IOU threshold for the NMS operation
    * `max_detections_per_class`: Maximum number of detections to retain per class.
    * `max_detections`: Maximum number of detections to retain across all classes.
    * `box_variance`: The scaling factors used to scale the bounding box predictions.
* OPTIONAL INPUTS: None
* OUTPUTS: a keras layer to decode predictions
* GLOBAL INPUTS: None



### data_funcs.py

---
##### random_flip_horizontal
```python
random_flip_horizontal(image, boxes)
```

Flips image and boxes horizontally with 50% chance

* INPUTS:
  * `image`: A 3-D tensor of shape `(height, width, channels)` representing an
    image.
  * `boxes`: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
    having normalized coordinates.
* OUTPUTS: Randomly flipped image and boxes

---
##### resize_and_pad_image
```python
resize_and_pad_image(image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0)
```

Resizes and pads image while preserving aspect ratio.
1. Resizes images so that the shorter side is equal to `min_side`
2. If the longer side is greater than `max_side`, then resize the image
  with longer side equal to `max_side`
3. Pad with zeros on right and bottom to make the image shape divisible by
`stride`

* INPUTS:
  * `image`: A 3-D tensor of shape `(height, width, channels)` representing an
    image.
  * `min_side`: The shorter side of the image is resized to this value, if
    `jitter` is set to None.
  * `max_side`: If the longer side of the image exceeds this value after
    resizing, the image is resized such that the longer side now equals to
    this value.
  * `jitter`: A list of floats containing minimum and maximum size for scale
    jittering. If available, the shorter side of the image will be
    resized to a random value in this range.
  * `stride`: The stride of the smallest feature map in the feature pyramid.
    Can be calculated using `image_size / feature_map_size`.
* OUTPUTS:
  `image`: Resized and padded image.
  `image_shape`: Shape of the image before padding.
  `ratio`: The scaling factor used to resize the image


---
##### preprocess_secoora_data
```python
preprocess_secoora_data(example)
```

This function preprocesses a secoora dataset for training

* INPUTS:
    * `val_dataset` [tensorflow dataset]: validation dataset
    * `train_dataset` [tensorflow dataset]: training dataset
* OPTIONAL INPUTS: None
* OUTPUTS:
    * `val_dataset` [tensorflow dataset]: validation dataset
    * `train_dataset` [tensorflow dataset]: training dataset
* GLOBAL INPUTS: None


---
##### preprocess_coco_data
```python
preprocess_coco_data(sample)
```

Applies preprocessing step to a single sample
* INPUTS:
  * `sample`: A dict representing a single training sample.
  * OPTIONAL INPUTS: None
* OUTPUTS:
  * `image`: Resized and padded image with random horizontal flipping applied.
  * `bbox`: Bounding boxes with the shape `(num_objects, 4)` where each box is
    of the format `[x, y, width, height]`.
  * `class_id`: An tensor representing the class id of the objects, having
    shape `(num_objects,)`.


---
##### swap_xy
```python
swap_xy(boxes)
```

Swaps order the of x and y coordinates of the boxes.
* INPUTS:
  `boxes`: A tensor with shape `(num_boxes, 4)` representing bounding boxes.
* OUTPUTS: swapped boxes with shape same as that of boxes.

---
##### convert_to_xywh
```python
convert_to_xywh(boxes)
```

Changes the box format to center, width and height.
* INPUTS:
  * `boxes`: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
    representing bounding boxes where each box is of the format
    `[xmin, ymin, xmax, ymax]`.
* OUTPUTS: converted boxes with shape same as that of boxes.

---
##### convert_to_corners
```python
convert_to_corners(boxes)
```

Changes the box format to corner coordinates
* INPUTS:
  * `boxes`: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)` representing bounding boxes where each box is of the format `[x, y, width, height]`.
* OUTPUTS: converted boxes with shape same as that of boxes.

---
##### compute_iou
```python
compute_iou(boxes1, boxes2)
```

This function computes pairwise IOU matrix for given two sets of boxes
* INPUTS:
    * `boxes1`: A tensor with shape `(N, 4)` representing bounding boxes
      where each box is of the format `[x, y, width, height]`.
    * `boxes2`: A tensor with shape `(M, 4)` representing bounding boxes
      where each box is of the format `[x, y, width, height]`.
* OPTIONAL INPUTS: None
* OUTPUTS:pairwise IOU matrix with shape `(N, M)`, where the value at ith row jth column holds the IOU between ith box and jth box from `boxes1` and `boxes2` respectively.
* GLOBAL INPUTS: None


##### AnchorBox
```python
AnchorBox()
```

Code from https://keras.io/examples/vision/retinanet/. Generates anchor boxes.
This class has operations to generate anchor boxes for feature maps at strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the format `[x, y, width, height]`.

* INPUTS:
  * `aspect_ratios`: A list of float values representing the aspect ratios of
    the anchor boxes at each location on the feature map
  * `scales`: A list of float values representing the scale of the anchor boxes
    at each location on the feature map.
  * `num_anchors`: The number of anchor boxes at each location on feature map
  * `areas`: A list of float values representing the areas of the anchor
    boxes for each feature map in the feature pyramid.
  * `strides`: A list of float value representing the strides for each feature
    map in the feature pyramid.
* OPTIONAL INPUTS: None
* OUTPUTS: anchor boxes for all the feature maps, stacked as a single tensor with shape
    `(total_anchors, 4)`, when `AnchorBox._get_anchors()` is called
* GLOBAL INPUTS: None

---
##### LabelEncoderCoco
```python
LabelEncoderCoco()
```

Transforms the raw labels into targets for training.
This class has operations to generate targets for a batch of samples which
is made up of the input images, bounding boxes for the objects present and
their class ids.

* INPUTS:
  * `anchor_box`: Anchor box generator to encode the bounding boxes.
  * `box_variance`: The scaling factors used to scale the bounding box targets.


### tfrecords_funcs.py

#### TF-dataset creation

---
##### prepare_image
```python
prepare_image(image)
```

This function resizes and pads an image, and rescales for resnet

* INPUTS:
    * `image` [tensor array]
* OPTIONAL INPUTS: None
* OUTPUTS:
    * `image` [tensor array]
GLOBAL INPUTS: None

---
##### prepare_secoora_datasets_for_training
```python
prepare_secoora_datasets_for_training(data_path, val_filenames)
```

This function prepares train and validation datasets  by extracting features (images, bounding boxes, and class labels) then map to preprocess_secoora_data, then apply prefetch, padded batch and label encoder

* INPUTS:
    * `data_path` [string]: path to the tfrecords
    * `train_filenames` [string]: tfrecord filenames for training
    * `val_filenames` [string]: tfrecord filenames for validation
* OPTIONAL INPUTS: None
* OUTPUTS:
    * `val_dataset` [tensorflow dataset]: validation dataset
    * `train_dataset` [tensorflow dataset]: training dataset
* GLOBAL INPUTS: None

---
##### prepare_secoora_datasets_for_training
```python
prepare_secoora_datasets_for_training(data_path, train_filenames, val_filenames)
```

This function prepares train and validation datasets  by extracting features (images, bounding boxes, and class labels)
then map to preprocess_secoora_data, then apply prefetch, padded batch and label encoder

* INPUTS:
    * `data_path` [string]: path to the tfrecords
    * `train_filenames` [string]: tfrecord filenames for training
    * `val_filenames` [string]: tfrecord filenames for validation
* OPTIONAL INPUTS: None
* OUTPUTS:
    * `val_dataset` [tensorflow dataset]: validation dataset
    * `train_dataset` [tensorflow dataset]: training dataset
* GLOBAL INPUTS: None

---
##### LabelEncoderCoco
```python
prepare_coco_datasets_for_training(train_dataset, val_dataset)
```

This function prepares a coco dataset loaded from tfds into one trainable by the model

* INPUTS:
    * `val_dataset` [tensorflow dataset]: validation dataset
    * `train_dataset` [tensorflow dataset]: training dataset
* OPTIONAL INPUTS: None
* OUTPUTS:
    * `val_dataset` [tensorflow dataset]: validation dataset
    * `train_dataset` [tensorflow dataset]: training dataset
* GLOBAL INPUTS: BATCH_SIZE

---
##### file2tensor
```python
file2tensor(f)
```

This function reads a jpeg image from file into a cropped and resized tensor,
for use in prediction with a trained mobilenet or vgg model
(the imagery is standardized depending on target model framework)

* INPUTS:
    * `f` [string] file name of jpeg
* OPTIONAL INPUTS:
    * `model` = {'mobilenet' | 'vgg'}
* OUTPUTS:
    * `image` [tensor array]: unstandardized image
    * `im` [tensor array]: standardized image
* GLOBAL INPUTS: TARGET_SIZE



#### TFRecord reading

---
##### write_tfrecords
```python
write_tfrecords(output_path, image_dir, csv_input)
```

This function writes tfrecords to fisk

* INPUTS:
    * `image_dir` [string]: place where jpeg images are
    * `csv_input` [string]: csv file that contains the labels
    * `output_path` [string]: place to writes files to
* OPTIONAL INPUTS: None
* OUTPUTS: None (tfrecord files written to disk)
* GLOBAL INPUTS: BATCH_SIZE

---
##### class_text_to_int
```python
class_text_to_int(row_label)
```

This function converts the string 'person' into the number 1

* INPUTS:
    * `row_label` [string]: class label string
* OPTIONAL INPUTS: None
* OUTPUTS: 1 or None
* GLOBAL INPUTS: BATCH_SIZE

---
##### split
```python
split(df, group)
```

This function splits a pandas dataframe by a pandas group object to extract the label sets from each image for writing to tfrecords

* INPUTS:
    * `df` [pandas dataframe]
    * `group` [pandas dataframe group object]
* OPTIONAL INPUTS: None
* OUTPUTS: tuple of bboxes and classes per image
* GLOBAL INPUTS: BATCH_SIZE

---
##### create_tf_example_coco
```python
create_tf_example_coco(group, path)
```

This function creates an example tfrecord consisting of an image and label encoded as bytestrings. The jpeg image is read into a bytestring, and the bbox coordinates and classes are collated and
converted also.

* INPUTS:
    * `group` [pandas dataframe group object]
    * `path` [tensorflow dataset]: training dataset
* OPTIONAL INPUTS: None
* OUTPUTS:
    * `tf_example` [tf.train.Example object]
* GLOBAL INPUTS: BATCH_SIZE


### plot_funcs.py

---
##### plot_history
```python
plot_history(history, train_hist_fig)
```

This function plots the training history of a model
* INPUTS:
    * `history` [dict]: the output dictionary of the model.fit() process, i.e. history = model.fit(...)
    * `train_hist_fig` [string]: the filename where the plot will be printed
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: None (figure printed to file)

---
##### visualize_detections
```python
visualize_detections(image, boxes, classes, scores, counter, str_prefix, figsize=(7, 7), linewidth=1, color=[0, 0, 1])
```
This function allows for visualization of imagery and bounding boxes

* INPUTS:
    * `images` [ndarray]: batch of images
    * `boxes` [ndarray]: batch of bounding boxes per image
    * `classes` [list]: class strings
    * `scores` [list]: prediction scores
    * `str_prefix` [string]: filename prefix
* OPTIONAL INPUTS:
  * `figsize`=(7, 7), figure size
  * `linewidth`=1, box line width
  * `color`=[0, 0, 1], box colour
* OUTPUTS:
    * `val_dataset` [tensorflow dataset]: validation dataset
    * `train_dataset` [tensorflow dataset]: training dataset
* GLOBAL INPUTS: None



## 3_ImageSeg

### General workflow using your own data

### model_funcs.py

#### Model creation

#### Model training


### tfrecords_funcs.py

#### TF-dataset creation

#### TFRecord reading


### plot_funcs.py





## 4_UnsupImageRecog

### General workflow using your own data


1. Create a TFREcord dataset from your data, organised as follows:
  * Copy training images into a folder called `train`
  * Copy validation images into a folder called `validation`
  * Ensure the class name is written to each file name. Ideally this is a prefix such that it is trivial to extract the class name from the file name
  * Modify one of the provided workflows (such as `tamucc_make_tfrecords.py`) for your dataset, to create your train and validation tfrecord shards

2. Set up your model
  * Decide on whether you want to train a small or large embedding model (`get_embedding_model` or `get_large_embedding_model`)

3. Set up a data pipeline
  * Modify and follow the provided examples to create a `get_training_dataset()` and `get_validation_dataset()`. This will likely require you copy and modify `get_batched_dataset` to your own needs, depending on the format of your labels in filenames, by writing your own `read_tfrecord` function for your dataset (depending on the model selected)
  * Remember for this method you have to read all the data at once into memory, which isn't ideal. You may therefore need to modify `get_data_stuff` to be a more efficient way to do so for your data

4. Set up a model training pipeline
  * `.compile()` your model with an appropriate loss function and metrics
  * Define a `LearningRateScheduler` function to vary learning rates over training as a function of training epoch
  * Define an `EarlyStopping` criteria and create a `ModelCheckpoint` to save trained model weights

5. Train the autoencoder embedding model
  * Use `history = model.fit()` to create a record of the training history. Pass the training and validation datasets, and a list of callbacks containing your model checkpoint, learning rate scheduler, and early stopping monitor)

6. Train the k-nearest neighbour classifer
  * Decide or determine the optimal number of neighbours (`k`)
  * Use `fit_knn_to_embeddings` to make a model of your training embeddings

6. Evaluate your model
  * Plot and study the `history` time-series of losses and metrics. If unsatisfactory, begin the iterative process of model optimization  
  * Use the `loss, accuracy = model.evaluate(get_validation_dataset(), batch_size=BATCH_SIZE, steps=validation_steps)` function using the validation dataset and specifying the number of validation steps
  * Make plots of model outputs, organized in such a way that you can at-a-glance see where the model is failing. Make use of `make_sample_plot` and `p_confmat`, as a starting point, to visualize sample imagery with their model predictions, and a confusion matrix of predicted/true class-correspondences
  * On the test set, play `tf.nn.l2_normalize` (i.e. don't use it on test and/or train embeddings and see if it improves results)



### model_funcs.py

#### Model creation

---
##### EmbeddingModel
```python
EmbeddingModel()
```

Code modified from https://keras.io/examples/vision/metric_learning/. This class allows an embedding model (an get_embedding_model or get_large_embedding_model instance)
to be trainable using the conventional model.fit(), whereby it can be passed another class
that provides batches of data examples in the form of anchors, positives, and negatives

* INPUTS: None
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: model training metrics

---
##### get_large_embedding_model
```python
get_large_embedding_model(TARGET_SIZE, num_classes, num_embed_dim)
```

Code modified from https://keras.io/examples/vision/metric_learning/. This function makes an instance of a larger embedding model, which is a keras sequential model
consisting of 5 convolutiional blocks, average 2d pooling, and an embedding layer

* INPUTS:
    * `model` [keras model]
    * `X_train` [list]
    * `ytrain` [list]
    * `num_dim_use` [int]
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * `knn` [sklearn knn model]


---
##### get_embedding_model
```python
get_embedding_model(TARGET_SIZE, num_classes, num_embed_dim)
```

Code modified from https://keras.io/examples/vision/metric_learning/. This function makes an instance of an embedding model, which is a keras sequential model
consisting of 3 convolutiional blocks, average 2d pooling, and an embedding layer

* INPUTS:
    * `model` [keras model]
    * `X_train` [list]
    * `ytrain` [list]
    * `num_dim_use` [int]
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * `knn` [sklearn knn model]


#### Model training

---
##### fit_knn_to_embeddings
```python
fit_knn_to_embeddings(model, X_train, ytrain, n_neighbors)
```

This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
saving out to the provided filename, `cm_filename`

* INPUTS:
    * `model` [keras model]
    * `X_train` [list]
    * `ytrain` [list]
    * `num_dim_use` [int]
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * `knn` [sklearn knn model]


---
##### weighted_binary_crossentropy
```python
weighted_binary_crossentropy(zero_weight, one_weight)
```

This function computes weighted binary crossentropy loss

* INPUTS:
    * `zero_weight` [float]: weight for the zero class
    * `one_weight` [float]: weight for the one class
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: the function `wbce` with all arguments passed

---
##### lrfn
```python
lrfn(epoch)
```
This function creates a custom piecewise linear-exponential learning rate function for a custom learning rate scheduler. It is linear to a max, then exponentially decays

* INPUTS: current `epoch` number
* OPTIONAL INPUTS: None
* GLOBAL INPUTS:`start_lr`, `min_lr`, `max_lr`, `rampup_epochs`, `sustain_epochs`, `exp_decay`
* OUTPUTS:  the function lr with all arguments passed


### tfrecords_funcs.py

#### TF-dataset creation

---
##### get_batched_dataset
```python
get_batched_dataset(filenames)
```

This function defines a workflow for the model to read data from
tfrecord files by defining the degree of parallelism, batch size, pre-fetching, etc
and also formats the imagery properly for model training
(assumes mobilenet by using read_tfrecord_mv2)

* INPUTS:
    * `filenames` [list]
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: BATCH_SIZE, AUTO
* OUTPUTS: `tf.data.Dataset` object


---
##### get_data_stuff
```python
get_data_stuff(ds, num_batches)
```

This function extracts lists of images and corresponding labels for training or testing

* INPUTS:
    * `ds` [PrefetchDataset]: either get_training_dataset() or get_validation_dataset()
    * `num_batches` [int]
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * `X` [list]
    * `y` [list]
    * `class_idx_to_train_idxs` [collections.defaultdict]

---
##### recompress_image
```python
recompress_image(image, label)
```

This function takes an image encoded as a byte string and recodes as an 8-bit jpeg. Label passes through unmodified

* INPUTS:
    * `image` [tensor array]
    * `label` [int]
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * `image` [tensor array]
    * `label` [int]

---
##### resize_and_crop_image
```python
resize_and_crop_image(image, label)
```

This function crops to square and resizes an image. The label passes through unmodified

* INPUTS:
    * `image` [tensor array]
    * `label` [int]
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: TARGET_SIZE
* OUTPUTS:
    * `image` [tensor array]
    * `label` [int]

---
##### to_tfrecord
```python
to_tfrecord(img_bytes, label, CLASSES)
```

This function creates a TFRecord example from an image byte string and a label feature

* INPUTS:
    * `img_bytes`
    * `label`
    * `CLASSES`
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: `tf.train.Feature` example


---
##### get_dataset_for_tfrecords
```python
get_dataset_for_tfrecords(recoded_dir, shared_size)
```

This function reads a list of TFREcord shard files, decode the images and label, resize and crop the image to `TARGET_SIZE` and creates batches

* INPUTS:
    * `recoded_dir`
    * `shared_size`
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: TARGET_SIZE
* OUTPUTS: `tf.data.Dataset` object

---
##### write_records
```python
write_records(tamucc_dataset, tfrecord_dir, CLASSES)
```

This function writes a `tf.data.Dataset` object to TFRecord shards

* INPUTS:
    * `tamucc_dataset` [tf.data.Dataset]
    * `tfrecord_dir` [string] : path to directory where files will be written
    * `CLASSES` [list] of class string names
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: None (files written to disk)


#### TFRecord reading


---
##### read_classes_from_json
```python
read_classes_from_json(json_file)
```

This function reads the contents of a json file enumerating classes

* INPUTS:
    * `json_file` [string]: full path to the json file
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: `CLASSES` [list]: list of classesd as byte strings

---
##### file2tensor
```python
file2tensor(f)
```

This function reads a jpeg image from file into a cropped and resized tensor, for use in prediction with a trained mobilenet or vgg model (the imagery is standardized depedning on target model framework)

* INPUTS:
    * `f` [string] file name of jpeg
* OPTIONAL INPUTS: None
* OUTPUTS:
    * `image` [tensor array]: unstandardized image
    * `im` [tensor array]: standardized image
* GLOBAL INPUTS: TARGET_SIZE


---
##### read_tfrecord
```python
read_tfrecord(example)
```

This function reads an example from a TFrecord file into a single image and label

* INPUTS:
    * TFRecord `example` object
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: TARGET_SIZE
* OUTPUTS:
    * `image` [tensor array]
    * `class_label` [tensor int]


---
##### read_image_and_label
```python
read_image_and_label(img_path)
```

This function reads a jpeg image from a provided filepath and extracts the label from the filename (assuming the class name is before `_IMG` in the filename)

* INPUTS:
    * `img_path` [string]: filepath to a jpeg image
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * `image` [tensor array]
    * `class_label` [tensor int]


### plot_funcs.py

---
##### conf_mat_filesamples
```python
conf_mat_filesamples(model, knn, sample_filenames, num_classes, num_dim_use, CLASSES)
```

This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
saving out to the provided filename, cm_filename

* INPUTS:
    * `model` [keras model]
    * `knn` [sklearn knn model]
    * `sample_filenames` [list] of strings
    * `num_classes` [int]
    * `num_dim_use` [int]
    * `CLASSES` [list] of strings: class names
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * `cm` [ndarray]: confusion matrix


---
##### p_confmat
```python
p_confmat(labs, preds, cm_filename, CLASSES, thres = 0.1)
```

This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
saving out to the provided filename, cm_filename

* INPUTS:
    * `labs` [ndarray]: 1d vector of labels
    * `preds` [ndarray]: 1d vector of model predicted labels
    * `cm_filename` [string]: filename to write the figure to
    * `CLASSES` [list] of strings: class names
* OPTIONAL INPUTS:
    * `thres` [float]: threshold controlling what values are displayed
* GLOBAL INPUTS: None
* OUTPUTS: None (figure printed to file)
