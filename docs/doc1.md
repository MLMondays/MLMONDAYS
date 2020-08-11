---
id: doc1
title: Documentation
sidebar_label: Overview
---

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
