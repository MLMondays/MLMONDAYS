---
id: doc4
title: ML MOndays API
---

Page under construction -- please check back later

This is the class and function reference of the ML Mondays course code

## 1_ImageRecog

### model_funcs.py

> lrfn(epoch)
This function creates a custom piecewise linear-exponential learning rate function
for a custom learning rate scheduler. It is linear to a max, then exponentially decays
INPUTS: current epoch number
OPTIONAL INPUTS: None
GLOBAL INPUTS: start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay
OUTPUTS:  the function lr with all arguments passed

> transfer_learning_model_vgg(num_classes, input_shape, dropout_rate=0.5)
This function creates an implementation of a convolutional deep learning model for estimating
a discrete category based on vgg, trained using transfer learning
(initialized using pretrained imagenet weights)
INPUTS:
    * num_classes = number of classes (output nodes on classification layer)
    * input_shape = size of input layer (i.e. image tensor)
OPTIONAL INPUTS:
    * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
GLOBAL INPUTS: None
OUTPUTS: keras model instance


> mobilenet_model(num_classes, input_shape, dropout_rate=0.5)
This function creates an implementation of a convolutional deep learning model for estimating
a discrete category based on mobilenet, trained from scratch
INPUTS:
    * num_classes = number of classes (output nodes on classification layer)
    * input_shape = size of input layer (i.e. image tensor)
OPTIONAL INPUTS:
    * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
GLOBAL INPUTS: None
OUTPUTS: keras model instance


> transfer_learning_mobilenet_model(num_classes, input_shape, dropout_rate=0.5)
This function creates an implementation of a convolutional deep learning model for estimating
a discrete category based on mobilenet v2, trained using transfer learning
(initialized using pretrained imagenet weights)
INPUTS:
    * num_classes = number of classes (output nodes on classification layer)
    * input_shape = size of input layer (i.e. image tensor)
OPTIONAL INPUTS:
    * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
GLOBAL INPUTS: None
OUTPUTS: keras model instance


### tfrecords_funcs.py


### plot_funcs.py
