# DL-CDI2020
Materials for the USGS "Deep Learning for Image Classification and Segmentation" CDI workshop, Fall 2020

One module per week. All modules include slides and jupyter notebooks


## Pre-requisites

#### Introduction to Machine Learning for Image Classification

Slides and jupyter notebooks introducing Machine Learning for Image Classification

#### Introduction to Deep Learning for Image Classification

Slides and jupyter notebooks introducing Deep Learning for Image Classification


## Module 1: Supervised Image Recognition (Whole-Image Classification)

#### LULC Classification Using Transfer Learning and Fine-Tuning

We use a convolutional neural network, with weights pre-trained on a large amount of imagery, then fine-tune to another dataset

#### LULC Classification Using Transfer Learning, Fine-Tuning, and Pruning 

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

We use a "human-in-the-loop" machine learning tool (based on a fully connected Conditional Random Field) to segment imagery based on sparse manual annotations. This can be used as a stand-alone tool, or to generate label imagery to train a deep learning based image segmentation model

#### LULC Segmentation using U-Nets

We use a deep convolutional "encoder-decoder" architecture called a "U-Net" for semantic segmentation of imagery


## Module 4: Unsupervised and Semi-Supervised Image Recognition

#### LULC Classification Using Distance Metric Learning (semi-supervised)

We use a deep convolutional "autoencoder" neural network to extract features from imagery, and then using a weakly supervised training strategy to create embedded features. Finally, we classify images based on how similar its embedded feature is to its nearest neighbors. An image is classified according to which class its features are closest to.

#### LULC Classification Using Deep Belief Networks (unsupervised)



