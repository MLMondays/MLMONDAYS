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

## TAMUCC
from tamucc_imports import *

##UNCOMMENT BELOW TO USE NWPU DATA
# from nwpu_imports import *

#see mlmondays blog post:
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

##calcs
import tensorflow as tf #numerical operations on gpu
import numpy as np #numerical operations on cpu

##plots
import matplotlib.pyplot as plt #for plotting
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix #compute confusion matrix from vectors of observed and estimated labels
import seaborn as sns #extended functionality / style to matplotlib plots
from matplotlib.offsetbox import OffsetImage, AnnotationBbox #for visualizing image thumbnails plotted as markers


SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

import tensorflow.keras.backend as K

from tfrecords_funcs import file2tensor
import tensorflow as tf #numerical operations on gpu



###############################################################
### PLOT FUNCTIONS
###############################################################

#-----------------------------------
def conf_mat_filesamples(model, knn, sample_filenames, num_classes, num_dim_use, CLASSES):
    """
    conf_mat_filesamples(model, knn, sample_filenames, num_classes, num_dim_use, CLASSES)
    This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
    using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
    saving out to the provided filename, cm_filename
    INPUTS:
        * model [keras model]
        * knn [sklearn knn model]
        * sample_filenames [list] of strings
        * num_classes [int]
        * num_dim_use [int]
        * CLASSES [list] of strings: class names
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * cm [ndarray]: confusion matrix
    """
    ## compute confusion matric for all samples
    cm = np.zeros((num_classes, num_classes))

    y_obs = []
    y_est = []
    for f in sample_filenames:
        # read image and convert to 32-bit tensor
        image = tf.cast(file2tensor(f), np.float32)
        # get the embeddings from the neural network model
        embeddings_sample = model.predict(tf.expand_dims(image, 0))
        # get class numeric code prediction from the k-nearest neighbours model
        est_class_idx = knn.predict(embeddings_sample[:,:num_dim_use])[0]
        y_est.append(est_class_idx)
        obs_class = f.split('/')[-1].split('_IMG')[0] #could this be a lambda function passed to the function as an argument?
        # get numeric code from class name
        class_idx = [i for i,c in enumerate(CLASSES) if c.decode()==obs_class][0]
        y_obs.append(class_idx)
        cm[class_idx, est_class_idx] += 1

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


#-----------------------------------
def p_confmat(labs, preds, cm_filename, CLASSES, thres = 0.1):
    """
    p_confmat(labs, preds, cm_filename, CLASSES, thres = 0.1)
    This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
    using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
    saving out to the provided filename, cm_filename
    INPUTS:
        * labs [ndarray]: 1d vector of labels
        * preds [ndarray]: 1d vector of model predicted labels
        * cm_filename [string]: filename to write the figure to
        * CLASSES [list] of strings: class names
    OPTIONAL INPUTS:
        * thres [float]: threshold controlling what values are displayed
    GLOBAL INPUTS: None
    OUTPUTS: None (figure printed to file)
    """
    cm = confusion_matrix(labs, preds)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cm[cm<thres] = 0

    plt.figure(figsize=(15,15))
    # sns.heatmap(cm,
    #     annot=True,
    #     cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True))
    plt.imshow(cm, cmap=plt.cm.hot)

    tick_marks = np.arange(len(CLASSES))+.5
    plt.xticks(tick_marks, [c.decode() for c in CLASSES], rotation=90,fontsize=12)
    plt.yticks(tick_marks, [c.decode() for c in CLASSES],rotation=0, fontsize=12)

    # plt.show()
    plt.savefig(cm_filename,
                dpi=200, bbox_inches='tight')
    plt.close('all')

    print("Average true positive rate across %i classes: %.3f" % (len(CLASSES), np.mean(np.diag(cm))))
