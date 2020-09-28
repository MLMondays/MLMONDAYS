---
id: doc1
title: Documentation
sidebar_label: Overview
---

ML-Mondays consists of 4 in-person classes, on Oct 5, Oct 13 (a day delayed, due to the Federal Holiday Columbus Day), Oct 19, and Oct 26. Each class follows on from the last. Classes 1 and 4 are pairs, as are classes 2 and 3. Participants are therefore expected to last the course. Optional homework assignments will be set for participants to carry out in their own time.

However, all course materials, including code, data, notebooks, this website, and videos, will be made available to the entire USGS in November, after the event. Full agenda to be announced later.

## Required pre-course reading

![](assets/phd.png)

Martin Gorner's 123 min (approx.) course called [TensorFlow, Keras and deep learning, without a PhD](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0) is a clear, approachable, fun introduction to neural networks. It is required pre-course reading for participants.


## Useful Links

### Reference

* [Github repository](https://github.com/dbuscombe-usgs/MLMONDAYS) (where all the code and website lives)

* [Documentation](https://dbuscombe-usgs.github.io/MLMONDAYS/docs/doc1)

* [API](https://dbuscombe-usgs.github.io/MLMONDAYS/docs/doc4) (list of all code functions, their inputs and outputs and what they do)

* [Summary of models](https://dbuscombe-usgs.github.io/MLMONDAYS/docs/doc1)

* [Blog](https://dbuscombe-usgs.github.io/MLMONDAYS/blog/)

### Notebooks

* [Part 1a jupyter notebook](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/1_ImageRecog/notebooks/MLMondays_week1_live_partA.ipynb)

* [Part 1a jupyter notebook for Colab](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/1_ImageRecog/notebooks/MLMondays_week1_live_partA_colab.ipynb)

* [Part 1b jupyter notebook](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/1_ImageRecog/notebooks/MLMondays_week1_live_partB.ipynb)

* [Part 2 jupyter notebook](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/2_ObjRecog/notebooks/MLMondays_week2_live.ipynb)

* [Part 2 jupyter notebook for Colab](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/2_ObjRecog/notebooks/MLMondays_week2_live_colab.ipynb)

* [Part 4 jupyter notebook](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/1_ImageUnsupRecog/notebooks/MLMondays_week4_live.ipynb)

* [Part 4 jupyter notebook for Colab](https://github.com/dbuscombe-usgs/MLMONDAYS/blob/master/1_ImageUnsupRecog/notebooks/MLMondays_week4_live_colab.ipynb)

### Data

* [Part 1 datasets repository](https://github.com/dbuscombe-usgs/mlmondays_data_imrecog)

* [Part 2 datasets repository](https://github.com/dbuscombe-usgs/mlmondays_data_objrecog)

* [Part 3 datasets repository](https://github.com/dbuscombe-usgs/mlmondays_data_imseg)

* [Part 4 datasets repository](https://github.com/dbuscombe-usgs/mlmondays_data_ssimrecog)


## Suggested pre-requisites

To gain more familiarity with machine learning and deep learning concepts and terminology, I recommend the following resources:

* a [great blog](https://blog.zenggyu.com/en/post/2018-12-05/retinanet-explained-and-demystified/) on the RetinaNet model for object detection
* a [visual introduction](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/) to machine learning
* a recent [review](https://arxiv.org/abs/2006.13311) of machine learning in geoscience
* this recent deep learning [review](https://dennybritz.com/blog/deep-learning-most-important-ideas/) and this reading [roadmap](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap)
* Daniel Worrall's excellent introductory lecture to machine learning [video](https://www.youtube.com/watch?v=FrbWQDdGpHQ&feature=youtu.be&t=40) and [slides](https://deworrall92.github.io/docs/MLSSIndo1_lo_res.pdf)
* Daniel Worrall's introductory lecture to machine learning [video](https://www.youtube.com/watch?v=K59cmobQKew&feature=youtu.be&t=270) and [slides](https://deworrall92.github.io/docs/MLSSIndo2_lo_res.pdf)
* a [list of resources](https://www.notion.so/fd42b6a13305452ba17a5e2fa71467a2?v=7d56617d132e4ec3b98121ae1070f024) for machine learning application in remote sensing
* a [blog](https://sayak.dev/tf.keras/data_augmentation/image/2020/05/10/augmemtation-recipes.html) on different tensorflow/keras data data augmentation recipes
* an overview of [gradient descent](https://ruder.io/optimizing-gradient-descent/)


(note: in the following `recognition` and `segmentation` are terms that imply specific forms of the more general term, `classification`)

## Week 1: Supervised Image Recognition

*A) Live session*: we'll work through jupyter notebooks containing workflows for image recognition (whole image classification). We'll be trying to answer the question, `How much of the Texas coastline is developed?`. To answer this question, we will train a deep learning model to classify aerial (oblique) images of the Texas coast, categorized into several natural and non-natural landuse/cover classes. See the [data page](doc2#how-much-of-the-texas-coastline-is-developed) for more information on the dataset.

#### Data Visualization

We're going to spend some time visualizing data, to introduce some techniques to do so that might be helpful in designing your own class labels for your own datasets. This is a practical class!

The visualizations will include mean images per class, per-channel, per-class histograms of image values, and dimensionality reduction techniques that might help both visualize and inform categorization of imagery. We'll see that the class boundaries are not at all distinct for the TAMUCC dataset - class boundaries are not distinct in terms of easily extractable image features. This exercise should convince you of the need for a more powerful supervised model

Finally, we'll see how the class boundaries in the NWPU dataset are better visualized using an unsupervised dimensionality reduction approach

#### Image recognition model training workflow

In this lesson, we will train a neural network 'end to end' in an extremely discriminative approach that explicitly maps the classes to the image features, and optimized to extract the features that explicitly predict the class. The network works by linking an image feature extractor to a classifying head, such that feature extraction is limited to only those that help predict the class. The feature extraction therefore results in classification directly.

For datasets where classes are obviously distinct, this is an extremely successful approach. We will see this with the NWPU dataset. However, for the TAMUCC dataset, where there is a lot more variability within classes and a lot less variability within classes, we will see how successful this approach is.


1. Set up a data workflow to feed the model as it trains
  * use batches fed optimally to the GPU from TFRecord files
  * use data augmentation as a regularization strategy
  * split into train (40% of the data) and validation portions (60%)

2. Train it
  * use transfer learning to train a classifier
  * use a learning rate scheduler to pass variable learning rates to the model as it trains
  * use 'early stopping' to base cessation of training on observed plateauing of validation loss
  * use a checkpoint to monitor validation loss and save the best model weights when the validation loss improves
  * use class weightings to lessen effects of class imbalance

3. Evaluate it
  * study the model training history - the loss and accuracy curves of train and validation sets
  * evaluate the performance of the trained model on the validation set
  * plot a 'confusion matrix' of correspondences between actual and estimate class labels
  * read some sample images from file, and use the model for prediction

4. Look at results from a  similar workflow on different class subsets


*B) Optional class assignment*: an additional dataset will be provided that you can work on using the same models introduced in the class. The [NWPU-RESISC45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html) is a publicly available benchmark for REmote Sensing Image Scene Classification (RESISC), created by Northwestern Polytechnical University (NWPU). This dataset contains 31,500 images, covering 45 scene classes with 700 images in each class. Participants will also be encouraged to adapt what they learned in the class to their own image recognition problems using their own data.


## Week 2: Supervised Image Object Detection

*A) Live session*: We'll work through jupyter notebooks containing workflows for image object detection (pixelwise classification). We'll be trying to answer the question, `How do people use beaches?`. To answer this question, we will train a deep learning model to detect and locate people in webcam (static, oblique) images of a beach in Florida. See the [data page](doc2#how-do-people-use-beaches) for more information.

#### RetinaNet
We are going to build a model called RetinaNet, to detect people in images of often-crowded beaches. RetinaNet is a popular single-stage object detector, which is accurate and runs fast. It uses a feature pyramid network to efficiently detect objects at multiple scales and uses a new Focal loss function, to alleviate the problem of the extreme foreground-background class imbalance.

* [Retina paper](https://arxiv.org/abs/1708.02002) Lin, T.Y., Goyal, P., Girshick, R., He, K. and Dollár, P., 2017. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision (pp. 2980-2988).
* [Feature pyramid paper](https://arxiv.org/abs/1612.03144)

RetinaNet adopts the Feature Pyramid Network (FPN) proposed by Lin et al. (2017) as its backbone, which is in turn built on top of ResNet-50 in a fully convolutional fashion. The fully convolutional nature enables the network to take an image of an arbitrary size and outputs proportionally sized feature maps at multiple levels in the feature pyramid.

See this excellent [blog post](https://blog.zenggyu.com/en/post/2018-12-05/retinanet-explained-and-demystified/) for more information

#### Fine-tune with COCO2017 subset
First, we'll initialize the model with COCO2017 weights for a model trained on the COCO2017 [data set](https://cocodataset.org/#overview) consisting of 164,000 images and  897,000  annotated  objects  from 80 categories. The 80 classes include 'person', which is also what we have labeled the SECOORA imagery. The other 79 classes include various types of vehicles, animals, sports, kitchen items, food items, personal accessories, furniture, etc. In other words, objects generally at close scale. In the imagery of beaches, or in imagery of natural environments, the objects of interest may not be so close-up. So, one of the attractive features of RetinaNet is the image pyramiding that detects object at 5 different scales. This implementation is based heavily on code [here](https://keras.io/examples/vision/retinanet/). When we ran `download_data.py` earlier on, we downloaded the weights and a subset of the COCO imagery [here](https://github.com/srihari-humbarwadi/datasets/releases)

Next, we'll fine-tune it on the SECOORA imagery, to illustrate the process of fine-tuning a model on similar dataset. We'll adopt a similar workflow to that in Part 1, where we used a learning rate scheduler, early stopping, and model checkpoints

#### Used the model to predict unseen imagery
Finally, I'll demonstrate the process of how you would use the trained/fined tuned model on sample imagery (jpeg images in a local folder)

#### Train a model from scratch
Next we'll show how to train a model from scratch - this takes too much time, so we'll download the weights from this exercise, load them into the model, and finally use for prediction

In the end, we'll see our model trained from scratch is an excellent way to count people on beaches

*B) Optional class assignment*: participants will be encouraged to adapt what they learned in the class to their own object recognition problems using their own data.


## Week 3: Supervised Image Segmentation

*A) Live session*: We'll work through jupyter notebooks containing workflows for image segmentation (pixelwise classification). We'll be trying to answer the question, `How much sand is there in the Outer Banks?`. To answer this question, we will train a deep learning model to segment dry sand pixels in aerial (nadir) imagery of the Outer Banks in North Carolina. See the [data page](doc2#how-much-sand-is-there-in-the-outer-banks) for more information.

*B) Optional class assignment*: an additional dataset will be provided that you can work on using the same models introduced in the class on your own. This [dataset](https://scholars.duke.edu/display/pub1419444) consists of aerial UAV colour imagery and labels of oyster reefs in shallow water, made publicly available by Duke University researcher [Patrick Gray](https://github.com/patrickcgray/oyster_net). There are two labels: `reef` and `no reef`. Participants will be encouraged to adapt what they learned in the class to their own image segmentation problems using their own data.

## Week 4: Semi-supervised Image Recognition

*A) Live session*: We'll work through jupyter notebooks containing workflows for more advanced and cutting edge *semi-supervised* methods for image recognition (whole image classification). We'll revisit the question posed in week 1, `How much of the Texas coastline is developed?`. See the [data page](doc2#how-much-of-the-texas-coastline-is-developed) for more information on the dataset. To answer this question, we will train a deep learning model to classify aerial (oblique) images of the Texas coast, categorized into several natural and non-natural landuse/cover classes. This time, however, we will use a different form of model that quantifies not only what class an image is in, but also a metric reporting close that is to the other classes. Training will utilize `soft` rather than `hard` labeling - a concept explained in the class - which is a potential strategy for dealing with small training datasets.

1. Set up a data workflow to feed the model as it trains
  * use batches fed optimally to the GPU from TFRecord files
  * this time we have to read imagery into memory, because the model needs to be fed small batches of anchor and positive examples, but access to all negative examples too
  * split into train (40% of the data) and validation portions (60%)

2. Train a "weakly supervised" image embedding model
  * the model is an autoencoder that creates an embedding feature for each image, such that that feature is maximally distant from all other features extracted from other classes
  * use a constant learning rate (a scheduler doesn't result in better results; this model is more stable with a constant learning rate, which becomes an important tunable hyperparameter)
  * use 'early stopping' to base cessation of training on observed plateauing of validation loss
  * use a checkpoint to monitor validation loss and save the best model weights when the validation loss improves

3. Evaluate the feature extractor
  * study the model training history - the loss and accuracy curves of train and validation sets

4. Construct an "unsupervised" classification model
  * build a k-nearest-neighbour (kNN) classifier that classifies unseen imagery based on the k nearest neighbours to the current image. Or, more correctly, the nearest neighbour's of the image's embedding vector in the training set of embedding vectors

5. Evaluate the classifier
  * evaluate the performance of the trained model on the validation set
  * plot a 'confusion matrix' of correspondences between actual and estimate class labels
  * read some sample images from file, and use the model for prediction

4. Fine-tune the model and evaluate it

5. Take a look at the same workflow for the NWPU data

*B) Optional class assignment*: participants will be encouraged to adapt what they learned in the class to their own semi-supervised image recognition problems using their own data.
