---
id: doc3
title: Models
---


## Image recognition


![](assets/imrecog_summary.png)

![](assets/imrecog_training.png)

![](assets/imrecog_prediction.png)

## Object recognition

![](assets/objrecog_summary.png)

RetinaNet is a popular single-stage object detector, which is accurate and runs fast. It uses a feature pyramid network to efficiently detect objects at multiple scales and uses a new Focal loss function, to alleviate the problem of the extreme foreground-background class imbalance.

![](assets/objrecog_training.png)

![](assets/objrecog_prediction.png)

## Image Segmentation

![](assets/imseg_summary.png)

Introduced in 2015, the U-Net model is still popular and is also commonly seen embedded in more complex deep learning models

The U-Net model is a simple fully  convolutional neural network that is used for binary segmentation i.e foreground and background pixel-wise classification. Mainly, it consists of two parts.

*  Encoder: we apply a series of conv layers and downsampling layers  (max-pooling) layers to reduce the spatial size
*  Decoder: we apply a series of upsampling layers to reconstruct the spatial size of the input.

The two parts are connected using a concatenation layers among different levels. This allows learning different features at different levels. At the end we have a simple conv 1x1 layer to reduce the number of channels to 1.

U-Net is symmetrical (hence the "U" in the name) and uses concatenation instead of addition to merge feature maps


![](assets/imseg_training.png)

The encoder (left hand side of the U) downsamples the N  x N x 3 image progressively using six banks of convolutional filters, each using filters double in size to the previous, thereby progressively downsampling the inputs as features are extracted through max pooling

A 'bottleneck' is just machine learning jargon for a very low-dimensional feature representation of a high dimensional input. Or, a relatively small vector of numbers that distill the essential information about a large image
An input of N x N x 3 (>>100,000 numbers) has been distilled to a 'bottleneck' of 16 x 16 x M (<<100,000 numbers)

![](assets/imseg_prediction.png)

The decoder (the right hand side of the U) upsamples the bottleneck into a N  x N x 1 label image progressively using six banks of convolutional filters, each using filters half in size to the previous, thereby progressively upsampling the inputs as features are extracted through transpose convolutions and concatenation. A transposed convolution is a relatively new type of deep learning model layer that convolves a dilated version of the input tensor, in order to upscale the output. The dilation operation consists of interleaving zeroed rows and columns between each pair of adjacent rows and columns in the input tensor. The dilation rate is the stride length

Finally, make the classification layer using one final convolutional layers that essentially just maps (by squishing over 16 layers) the output of the previous layer to a single 2D output (with values ranging from 0 to 1) based on a sigmoid activation function

## Semi-supervised image recognition

![](assets/ssimrecog_summary.png)

![](assets/ssimrecog_training.png)

![](assets/ssimrecog_prediction.png)

![](assets/ssimrecog_prediction2.png)
