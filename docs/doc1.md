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

## Suggested pre-requisites

To gain more familiarity with machine learning and deep learning concepts and terminology, I recommend the following resources:

* a [visual introduction](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/) to machine learning
* a recent [review](https://arxiv.org/abs/2006.13311) of machine learning in geoscience
* this recent deep learning [review](https://dennybritz.com/blog/deep-learning-most-important-ideas/) and this reading [roadmap](https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap)
* Daniel Worrall's excellent introductory lecture to machine learning [video](https://www.youtube.com/watch?v=FrbWQDdGpHQ&feature=youtu.be&t=40) and [slides](https://deworrall92.github.io/docs/MLSSIndo1_lo_res.pdf)
* Daniel Worrall's introductory lecture to machine learning [video](https://www.youtube.com/watch?v=K59cmobQKew&feature=youtu.be&t=270) and [slides](https://deworrall92.github.io/docs/MLSSIndo2_lo_res.pdf)
* a [list of resources](https://www.notion.so/fd42b6a13305452ba17a5e2fa71467a2?v=7d56617d132e4ec3b98121ae1070f024) for machine learning application in remote sensing


(note: in the following `recognition` and `segmentation` are terms that imply specific forms of the more general term, `classification`)

## Week 1: Supervised Image Recognition

*A) Live session*: we'll work through jupyter notebooks containing workflows for image recognition (whole image classification). We'll be trying to answer the question, `How much of the Texas coastline is developed?`. To answer this question, we will train a deep learning model to classify aerial (oblique) images of the Texas coast, categorized into several natural and non-natural landuse/cover classes. See the [data page](doc2#how-much-of-the-texas-coastline-is-developed) for more information on the dataset.

*B) Optional class assignment*: an additional dataset will be provided that you can work on using the same models introduced in the class. The [NWPU-RESISC45](http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html) is a publicly available benchmark for REmote Sensing Image Scene Classification (RESISC), created by Northwestern Polytechnical University (NWPU). This dataset contains 31,500 images, covering 45 scene classes with 700 images in each class. Participants will also be encouraged to adapt what they learned in the class to their own image recognition problems using their own data.

## Week 2: Supervised Image Object Detection

*A) Live session*: We'll work through jupyter notebooks containing workflows for image object detection (pixelwise classification). We'll be trying to answer the question, `How do people use beaches?`. To answer this question, we will train a deep learning model to detect and locate people in webcam (static, oblique) images of a beach in Florida. See the [data page](doc2#how-do-people-use-beaches) for more information.

*B) Optional class assignment*: participants will be encouraged to adapt what they learned in the class to their own object recognition problems using their own data.

## Week 3: Supervised Image Segmentation

*A) Live session*: We'll work through jupyter notebooks containing workflows for image segmentation (pixelwise classification). We'll be trying to answer the question, `How much sand is there in the Outer Banks?`. To answer this question, we will train a deep learning model to segment dry sand pixels in aerial (nadir) imagery of the Outer Banks in North Carolina. See the [data page](doc2#how-much-sand-is-there-in-the-outer-banks) for more information.

*B) Optional class assignment*: an additional dataset will be provided that you can work on using the same models introduced in the class on your own. This [dataset](https://scholars.duke.edu/display/pub1419444) consists of aerial UAV color imagery and labels of oyster reefs in shallow water, made publicly available by Duke University researcher [Patrick Gray](https://github.com/patrickcgray/oyster_net). There are two labels: `reef` and `no reef`. Participants will be encouraged to adapt what they learned in the class to their own image segmentation problems using their own data.

## Week 4: Semi-supervised Image Recognition

*A) Live session*: We'll work through jupyter notebooks containing workflows for more advanced and cutting edge *semi-supervised* methods for image recognition (whole image classification). We'll revisit the question posed in week 1, `How much of the Texas coastline is developed?`. See the [data page](doc2#how-much-of-the-texas-coastline-is-developed) for more information on the dataset. To answer this question, we will train a deep learning model to classify aerial (oblique) images of the Texas coast, categorized into several natural and non-natural landuse/cover classes. This time, however, we will use a different form of model that quantifies not only what class an image is in, but also a metric reporting close that is to the other classes. Training will utilize `soft` rather than `hard` labeling - a concept explained in the class - which is a potential strategy for dealing with small training datasets.

*B) Optional class assignment*: participants will be encouraged to adapt what they learned in the class to their own semi-supervised image recognition problems using their own data.


## General lesson structure

We will keep to a similar lesson format each week. It incorporates a generic workflow that I have researched that fits many use-cases, starting with a simple model, and building complexity as needed. Not all problems will need every step, but each step will be introduced in case you need to employ and adapt the workflow to your own needs/datasets. The generic lesson outline is below:

*1) Introduction*

Brief introduction to the topic; the types of problems it is designed to solve; the types of data that are suitable; and a brief tour of where this technique fits in the general workflow for the course, and the general machine learning landscape.

*2) Datasets*

Each week, we will have more than one dataset to play with. The purpose is primarily to show you subtly different workflows dictated by the specifics of each dataset (dealing with variety in file naming, folder names, labels, splits, image sizes, etc). It also gives you more opportunity to learn in your own time (outside and beyond the class). Each dataset will be described, and you will be pointed to workflows for each dataset - blog posts, code, files, etc.

*3) Visualization of Primary Dataset*

Due to time constraints, we will demonstrate each topic each week with only one - primary - dataset. This section will show you ways to visualize the data, from simple plots, to identifying outliers, removing corrupt files, and other utilities

*4) Train a simplified model, as demonstration*

We will always start with a relatively simple model. This would be conceptually similar to a larger, more powerful model that we would subsequently employ. It should be - at least by the end of the course - intuitive how this model is put together. It would typically take a relatively short time to train. It is used to demonstrate a general workflow that you would adapt to larger, more complicated models.

*5) Train a more complex model, using transfer learning*

The deep neural models we will use will of course vary, but in general they consist of large image 'feature extractors' (explanation forthcoming) and a classifying layer. We'll demonstrate a more complex workflow, utilizing larger models and weights learned on different data sets. This will demonstrate the principle of `transfer learning`, which is inheriting model weights learned from the same model framework on a different image dataset, then adapting that model to your dataset by leaving the feature extractor alone, and building on top another set of model layers ending in a new classifying layer adapted to your data/classes.

*6) Train a more complex model, either from scratch or using fine-tuning*

This part of the workflow won't always be necessary to achieve a certain desired level of model accuracy. But would certainly be helpful for particularly difficult tasks. This version of the model does not employ transfer learning - instead it is either trained from scratch (all model weights initialized with random numbers and learned from scratch) or fine-tuned, which is using transfer learning but allowing the weights in the feature extractor to continue to modify based on further training of the model.

*7) Lecture*

Deep learning models can take a long time to train, so an efficient use of our time while this occurs is a traditional lecture slideshow format where we can go over some of the theoretical details, some of the broader statistical and computer science themes we are utilizing, and some words on datasets, and data processing pipelines

*8) Model evaluation*

When the model is trained, we will evaluate it in a variety of ways using unseen test data

*9) Model optimization*

You may never be 'done' training and optimizing your model - there are so many things to tweak! The standard list includes, in order of usual importance:

* learning rate, and learning rate schedulers
* other model hyperparameters (such as dropout rate, and batch size)
* model layers (such as residual connections and batch normalization)
* data augmentation (we'll talk about what this means, but essentially it is a way to train with more data generated from your existing data, *augmenting* your existing dataset)

*10) Train an optimized model, with cross-validation*

When you are finally happy with your model, the final training procedure should ideally use cross-validation to better understand its sensitivity to varying inputs (random order of training and validation subsets). The various models can also be used as ensembles. We'll briefly show how to ensemble any model.
