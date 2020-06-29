# DL-CDI2020
Materials for the USGS "Deep Learning for Image Classification and Segmentation" CDI workshop, Fall 2020

One module per week. All modules include slides, videos and jupyter notebooks. Each module introduce two topics. A third (more advanced) topic may be introduced using video


## Pre-requisites

#### Introduction to Machine Learning for Image Classification

Slides and jupyter notebooks introducing Machine Learning for Image Classification

#### Introduction to Deep Learning for Image Classification

Slides and jupyter notebooks introducing Deep Learning for Image Classification


## Module 1: Supervised Image Recognition (Whole-Image Classification)

#### LULC Classification Using Transfer Learning and Fine-Tuning

We use a convolutional neural network, with weights pre-trained on a large amount of imagery, then fine-tune to another dataset

#### LULC Classification Using Transfer Learning, Fine-Tuning, and Pruning (video)

We repeat the previous exercise, this time using model pruning to acheive a more parsimonious predictor

#### LULC Classification From Scratch

We use a convolutional neural network, with weights trained from scratch with no transfer learning


## Module 2: Supervised Object Recognition (Bounding Box Classification)

#### Makesense.ai

Using www.makesense.ai to create a bounding box dataset of people on beaches. Using SECOORA webcam imagery.

#### Counting People on Beaches with YOLO

We use a deep neural network called YOLO, which is a single-shot-detector or SSD type model for efficient detection of objects in an image


## Module 3: Supervised Image Segmentation (Pixel Classification)

#### Semi-supervised image segmentation using Doodler

We use a "human-in-the-loop" machine learning tool called `Doodler` (authored by myself, based on a fully connected Conditional Random Field) to segment imagery based on sparse manual annotations. This can be used as a stand-alone tool, or to generate label imagery to train a deep learning based image segmentation model

#### Binary segmentation of intertidal reefs using U-Nets

We use a deep convolutional "encoder-decoder" architecture called a "U-Net" for semantic segmentation of imagery

#### Multiclass segmentation of LULC using U-Nets (video)
We use a different U-Net model trained for each class, and merge them into a multiclass label image


## Module 4: Semi-Supervised and Unsupervised Image Recognition

#### LULC Classification Using Distance Metric Learning (semi-supervised)

We use a deep convolutional "autoencoder" neural network to extract features from imagery, and then using a weakly supervised training strategy to create embedded features. Finally, we classify images based on how similar its embedded feature is to its nearest neighbors. An image is classified according to which class its features are closest to.

#### LULC Classification Using Deep Belief Networks (unsupervised and semi-supervised)

We use a Deep Belief Network (DBN) to classify imagery using unsupervised training. The DBN consists of several RBNs or Restricted Boltzman Machines, which are shallow neural networks for feature extraction. We use the DBN to learn the underlying structure in the data (unsupervised), and classify using that. Then we use some labels to fine-tune the DBN, and classify, which is considered 'semi-supervised'. 

#### Using Deep Belief Networks for training with augmented data (video)

The DBN is a generative model that is able to generate 'fake' realizations of the data. We can use this to generate many more examples of each class, and train a classifier with this augmented data set


## Datasets

#### Prerequisites: 
TBD

#### Modules 1 and 4: 
* NWPU-45
* EuroSAT
* Our own coastal classes

#### Module 2: 
* SECOORA webcam videos

#### Module 3: 
* OysterNet
* Our own coastal classes

