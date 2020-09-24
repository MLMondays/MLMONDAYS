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

## TAMUCC DATA imports
from tamucc_imports import *

##UNCOMMENT BELOW TO USE NWPU DATA
# from nwpu_imports import *

# #see mlmondays blog post:
# import os
# os.environ["TF_DETERMINISTIC_OPS"] = "1"

from tfrecords_funcs import *

from model_funcs import *

from plot_funcs import *

#
# ##calcs
# import tensorflow as tf #numerical operations on gpu
# # from tensorflow.keras.applications import MobileNetV2 #mobilenet v2 model, used for feature extraction
# # from tensorflow.keras.applications import VGG16 #vgg model, used for feature extraction
# # from tensorflow.keras.applications import Xception #xception model, used for feature extraction
# import numpy as np #numerical operations on cpu
# # from sklearn.decomposition import PCA  #for data dimensionality reduction / viz.
# # from sklearn.preprocessing import StandardScaler #data scaling data in PCA and TSNE algorithms
# # from sklearn.manifold import TSNE #for data dimensionality reduction / viz.
#
# # ## plots
# # import matplotlib.pyplot as plt #for plotting
# # from sklearn.metrics import confusion_matrix #compute confusion matrix from vectors of observed and estimated labels
# # import seaborn as sns #extended functionality / style to matplotlib plots
# # from matplotlib.offsetbox import OffsetImage, AnnotationBbox #for visualizing image thumbnails plotted as markers

##i/o
import pandas as pd #for data wrangling. We just use it to read csv files
import shutil #json for class file reading, shutil for file copying/moving

##utils
from sklearn.utils import class_weight #utility for computinh normalised class weights
#keras functions for early stopping and model weights saving
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K #access to keras backend functions
# from collections import defaultdict

# set a seed for reproducibility
SEED=42
np.random.seed(SEED)
tf.random.set_seed(SEED)

#for automatically determining dataset feeding processes based on available hardware
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
