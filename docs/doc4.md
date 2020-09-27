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
```python
transfer_learning_model_vgg(num_classes, input_shape, dropout_rate=0.5)
```

This function creates an implementation of a convolutional deep learning model for estimating
a discrete category based on vgg, trained using transfer learning
(initialized using pretrained imagenet weights)

* INPUTS:
    * num_classes = number of classes (output nodes on classification layer)
    * input_shape = size of input layer (i.e. image tensor)
* OPTIONAL INPUTS:
    * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
* GLOBAL INPUTS: None
* OUTPUTS: keras model instance

---
```python
mobilenet_model(num_classes, input_shape, dropout_rate=0.5)
```

This function creates an implementation of a convolutional deep learning model for estimating
a discrete category based on mobilenet, trained from scratch

* INPUTS:
    * num_classes = number of classes (output nodes on classification layer)
    * input_shape = size of input layer (i.e. image tensor)
* OPTIONAL INPUTS:
    * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
* GLOBAL INPUTS: None
* OUTPUTS: keras model instance

---
```python
transfer_learning_mobilenet_model(num_classes, input_shape, dropout_rate=0.5)
```

This function creates an implementation of a convolutional deep learning model for estimating
a discrete category based on mobilenet v2, trained using transfer learning
(initialized using pretrained imagenet weights)

* INPUTS:
    * num_classes = number of classes (output nodes on classification layer)
    * input_shape = size of input layer (i.e. image tensor)
* OPTIONAL INPUTS:
    * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
* GLOBAL INPUTS: None
* OUTPUTS: keras model instance


---
```python
transfer_learning_xception_model(num_classes, input_shape, dropout_rate=0.25)
```

This function creates an implementation of a convolutional deep learning model for estimating
a discrete category based on xception, trained using transfer learning
(initialized using pretrained imagenet weights)

* INPUTS:
    * num_classes = number of classes (output nodes on classification layer)
    * input_shape = size of input layer (i.e. image tensor)
* OPTIONAL INPUTS:
    * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
* GLOBAL INPUTS: None
* OUTPUTS: keras model instance


---
```python
xception_model(num_classes, input_shape, dropout_rate=0.25)
```

This function creates an implementation of a convolutional deep learning model for estimating
a discrete category based on xception, trained from scratch

* INPUTS:
    * num_classes = number of classes (output nodes on classification layer)
    * input_shape = size of input layer (i.e. image tensor)
* OPTIONAL INPUTS:
    * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
* GLOBAL INPUTS: None
* OUTPUTS: keras model instance

---
```python
conv_block(inp, filters=32, bn=True, pool=True)
```

This function generates a convolutional block

* INPUTS:
    * inp = input layer
* OPTIONAL INPUTS:
    * filters = number of convolutional filters to use
    * bn=False, use batch normalization in each convolutional layer
    * pool=True, use pooling in each convolutional layer
    * shallow=True, if False, a larger model with more convolution layers is used
* GLOBAL INPUTS: None
* OUTPUTS: keras model layer object

---
```python
make_cat_model(num_classes, dropout, denseunits, base_filters, bn=False, pool=True, shallow=True)
```

This function creates an implementation of a convolutional deep learning model for estimating
a discrete category

* INPUTS:
    * num_classes = number of classes (output nodes on classification layer)
    * dropout = proportion of neurons to randomly set to zero, after the pooling layer
    * denseunits = number of neurons in the classifying layer
    * base_filters = number of convolutional filters to use in the first layer
* OPTIONAL INPUTS:
    * bn=False, use batch normalization in each convolutional layer
    * pool=True, use pooling in each convolutional layer
    * shallow=True, if False, a larger model with more convolution layers is used
* GLOBAL INPUTS: TARGET_SIZE
* OUTPUTS: keras model instance

#### Model training

---
```python
lrfn(epoch)
```

This function creates a custom piecewise linear-exponential learning rate function for a custom learning rate scheduler. It is linear to a max, then exponentially decays

* INPUTS: current epoch number
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay
* OUTPUTS:  the function lr with all arguments passed


### tfrecords_funcs.py

#### TF-dataset creation

---
```python
get_batched_dataset(filenames)
```

This function defines a workflow for the model to read data from
tfrecord files by defining the degree of parallelism, batch size, pre-fetching, etc
and also formats the imagery properly for model training
(assumes mobilenet by using read_tfrecord_mv2)

* INPUTS:
    * filenames [list]
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: BATCH_SIZE, AUTO
* OUTPUTS: tf.data.Dataset object

---
```python
get_eval_dataset(filenames)
```

This function defines a workflow for the model to read data from
tfrecord files by defining the degree of parallelism, batch size, pre-fetching, etc
and also formats the imagery properly for model training
(assumes mobilenet by using read_tfrecord_mv2). This evaluation version does not .repeat() because it is not being called repeatedly by a model

* INPUTS:
    * filenames [list]
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: BATCH_SIZE, AUTO
* OUTPUTS: tf.data.Dataset object

---
```python
resize_and_crop_image(image, label)
```

This function crops to square and resizes an image. The label passes through unmodified

* INPUTS:
    * image [tensor array]
    * label [int]
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: TARGET_SIZE
* OUTPUTS:
    * image [tensor array]
    * label [int]

---
```python
recompress_image(image, label)
```

This function takes an image encoded as a byte string and recodes as an 8-bit jpeg. Label passes through unmodified

* INPUTS:
    * image [tensor array]
    * label [int]
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * image [tensor array]
    * label [int]


#### TFRecord reading

---
```python
file2tensor(f, model='mobilenet')
```

This function reads a jpeg image from file into a cropped and resized tensor,
for use in prediction with a trained mobilenet or vgg model
(the imagery is standardized depending on target model framework)

* INPUTS:
    * f [string] file name of jpeg
* OPTIONAL INPUTS:
    * model = {'mobilenet' | 'vgg'}
* OUTPUTS:
    * image [tensor array]: unstandardized image
    * im [tensor array]: standardized image
* GLOBAL INPUTS: TARGET_SIZE

---
```python
read_classes_from_json(json_file)
```

This function reads the contents of a json file enumerating classes

* INPUTS:
    * json_file [string]: full path to the json file
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * CLASSES [list]: list of classesd as byte strings

---
```python
read_tfrecord_vgg(example)
```

This function reads an example record from a tfrecord file
and parses into label and image ready for vgg model training

* INPUTS:
    * example: an tfrecord 'example' object, containing an image and label
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: TARGET_SIZE
* OUTPUTS:
    * image [tensor]: resized and pre-processed for vgg
    * class_label [tensor] 32-bit integer


---
```python
read_tfrecord_mv2(example)
```

This function reads an example record from a tfrecord file
and parses into label and image ready for mobilenet model training

* INPUTS:
    * example: an tfrecord 'example' object, containing an image and label
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: TARGET_SIZE
* OUTPUTS:
    * image [tensor]: resized and pre-processed for mobilenetv2
    * class_label [tensor] 32-bit integer


---
```python
read_tfrecord(example)
```

This function reads an example from a TFrecord file into a single image and label

* INPUTS:
    * TFRecord example object
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: TARGET_SIZE
* OUTPUTS:
    * image [tensor array]
    * class_label [tensor int]

---
```python
read_image_and_label(img_path)
```

This function reads a jpeg image from a provided filepath and extracts the label from the filename (assuming the class name is before "IMG" in the filename)

* INPUTS:
    * img_path [string]: filepath to a jpeg image
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * image [tensor array]
    * class_label [tensor int]


#### TFRecord creation

---
```python
get_dataset_for_tfrecords(recoded_dir, shared_size)
```

This function reads a list of TFREcord shard files, decode the images and label resize and crop the image to TARGET_SIZE, and create batches

* INPUTS:
    * recoded_dir
    * shared_size
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: TARGET_SIZE
* OUTPUTS:
    * tf.data.Dataset object

---
```python
write_records(tamucc_dataset, tfrecord_dir, CLASSES)
```

This function writes a tf.data.Dataset object to TFRecord shards

* INPUTS:
    * tamucc_dataset [tf.data.Dataset]
    * tfrecord_dir [string] : path to directory where files will be written
    * CLASSES [list] of class string names
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: None (files written to disk)

---
```python
to_tfrecord(img_bytes, label, CLASSES)
```

This function creates a TFRecord example from an image byte string and a label feature

* INPUTS:
    * img_bytes: an image bytestring
    * label: label string of image
    * CLASSES: list of string classes in the entire dataset
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: tf.train.Feature example


### plot_funcs.py

---
```python
plot_history(history, train_hist_fig)
```

This function plots the training history of a model

* INPUTS:
    * history [dict]: the output dictionary of the model.fit() process, i.e. history = model.fit(...)
    * train_hist_fig [string]: the filename where the plot will be printed
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: None (figure printed to file)

---
```python
get_label_pairs(val_ds, model)
```

This function gets label observations and model estimates

* INPUTS:
    * val_ds: a batched data set object
    * model: trained and compiled keras model instance
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * labs [ndarray]: 1d vector of numeric labels
    * preds [ndarray]: 1d vector of correspodning model predicted numeric labels

---
```python
p_confmat(labs, preds, cm_filename, CLASSES, thres = 0.1)
```

This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
saving out to the provided filename, cm_filename

* INPUTS:
    * labs [ndarray]: 1d vector of labels
    * preds [ndarray]: 1d vector of model predicted labels
    * cm_filename [string]: filename to write the figure to
    * CLASSES [list] of strings: class names
* OPTIONAL INPUTS:
    * thres [float]: threshold controlling what values are displayed
* GLOBAL INPUTS: None
* OUTPUTS: None (figure printed to file)

---
```python
make_sample_plot(model, sample_filenames, test_samples_fig, CLASSES))
```

This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
saving out to the provided filename, cm_filename

* INPUTS:
    * model: trained and compiled keras model
    * sample_filenames: [list] of strings
    * test_samples_fig [string]: filename to print figure to
    * CLASSES [list] os trings: class names
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: None (matplotlib figure, printed to file)

---
```python
compute_hist(images)
```

Compute the per channel histogram for a batch
of images

* INPUTS:
    * images [ndarray]: batch of shape (N x W x H x 3)
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS:
    * hist_r [dict]: histogram frequencies {'hist'} and bins {'bins'} for red channel
    * hist_g [dict]: histogram frequencies {'hist'} and bins {'bins'} for green channel
    * hist_b [dict]: histogram frequencies {'hist'} and bins {'bins'} for blue channel

---
```python
plot_distribution(images, labels, class_id, CLASSES)
```

Compute the per channel histogram for a batch of images

* INPUTS:
    * images [ndarray]: batch of shape (N x W x H x 3)
    * labels [ndarray]: batch of shape (N x 1)
    * class_id [int]: class integer to plot
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: None
* OUTPUTS: matplotlib figure

---
```python
plot_one_class(inp_batch, sample_idx, label, batch_size, CLASSES, rows=8, cols=8, size=(20,15))
```

Plot "batch_size" images that belong to the class "label"

* INPUTS:
    * inp_batch [ndarray]: batch of N images
    * sample_idx [list]: indices of the N images
    * label [string]: class string
    * batch_size [int]: number of images to plot
* OPTIONAL INPUTS:
    * rows=8 [int]: number of rows
    * cols=8 [int]: number of columns
    * size=(20,15) [tuple]: size of matplotlib figure
* GLOBAL INPUTS: None (matplotlib figure, printed to file)

---
```python
compute_mean_image(images, opt="mean")
```

Compute and return mean image given a batch of images

* INPUTS:
    * images [ndarray]: batch of shape (N x W x H x 3)
* OPTIONAL INPUTS:
    * opt="mean" or "median"
* GLOBAL INPUTS:
* OUTPUTS: 2d mean image [ndarray]

---
```python
plot_mean_images(images, labels, CLASSES, rows=3, cols = 2)
```

Plot the mean image of a set of images

* INPUTS:
    * images [ndarray]: batch of shape (N x W x H x 3)
    * labels [ndarray]: batch of shape (N x 1)
* OPTIONAL INPUTS:
  * rows [int]: number of rows
  * cols [int]: number of columns
* GLOBAL INPUTS: CLASSES
* OUTPUTS: matplotlib figure

---
```python
plot_tsne(tsne_result, label_ids, CLASSES)
```

Plot TSNE loadings and colour code by class. [Source](https://www.kaggle.com/gaborvecsei/plants-t-sne)

* INPUTS:
  * tsne_result [ndarray]: N x 2 data of loadings on two axes
  * label_ids [int]: N class labels
* OPTIONAL INPUTS: None
* GLOBAL INPUTS: CLASSES
* OUTPUTS: matplotlib figure, matplotlib figure axes object

---
```python
visualize_scatter_with_images(X_2d_data, labels, images, figsize=(15,15), image_zoom=1,xlim = (-3,3), ylim=(-3,3))
```

Plot TSNE loadings and colour code by class. [Source](https://www.kaggle.com/gaborvecsei/plants-t-sne)

* INPUTS:
    * X_2d_data [ndarray]: N x 2 data of loadings on two axes
    * images [ndarray] : N batch of images to plot
* OPTIONAL INPUTS:
    * figsize=(15,15)
    * image_zoom=1 [float]: control the scaling of the imagery (make smaller for smaller thumbnails)
    * xlim = (-3,3) [tuple]: set x axes limits
    * ylim = (-3,3) [tuple]: set y axes limits]
* GLOBAL INPUTS: None
* OUTPUTS: matplotlib figure



## 2_ObjRecog

### General workflow using your own data

### model_funcs.py

#### Model creation

#### Model training

### data_funcs.py


### tfrecords_funcs.py

#### TF-dataset creation

#### TFRecord reading


### plot_funcs.py





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

### model_funcs.py

#### Model creation

#### Model training


### tfrecords_funcs.py

#### TF-dataset creation

#### TFRecord reading


### plot_funcs.py
