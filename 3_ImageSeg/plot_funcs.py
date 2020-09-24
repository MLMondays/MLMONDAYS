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


from oyster_imports import *

from tfrecords_funcs import seg_file2tensor
import tensorflow as tf #numerical operations on gpu

#see mlmondays blog post:
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

##calcs
import numpy as np

SEED=42
np.random.seed(SEED)

import matplotlib.pyplot as plt


###############################################################
### PLOTTING FUNCTIONS
###############################################################

#-----------------------------------
def plot_seg_history_iou(history, train_hist_fig):
    """
    "plot_seg_history_iou"
    This function plots the training history of a model
    INPUTS:
        * history [dict]: the output dictionary of the model.fit() process, i.e. history = model.fit(...)
        * train_hist_fig [string]: the filename where the plot will be printed
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (figure printed to file)
    """
    n = len(history.history['val_loss'])

    plt.figure(figsize=(20,10))
    plt.subplot(121)
    plt.plot(np.arange(1,n+1), history.history['mean_iou'], 'b', label='train accuracy')
    plt.plot(np.arange(1,n+1), history.history['val_mean_iou'], 'k', label='validation accuracy')
    plt.xlabel('Epoch number', fontsize=10); plt.ylabel('Mean IoU Coefficient', fontsize=10)
    plt.legend(fontsize=10)

    plt.subplot(122)
    plt.plot(np.arange(1,n+1), history.history['loss'], 'b', label='train loss')
    plt.plot(np.arange(1,n+1), history.history['val_loss'], 'k', label='validation loss')
    plt.xlabel('Epoch number', fontsize=10); plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=10)

    # plt.show()
    plt.savefig(train_hist_fig, dpi=200, bbox_inches='tight')

#-----------------------------------
def plot_seg_history(history, train_hist_fig):
    """
    "plot_seg_history"
    This function plots the training history of a model
    INPUTS:
        * history [dict]: the output dictionary of the model.fit() process, i.e. history = model.fit(...)
        * train_hist_fig [string]: the filename where the plot will be printed
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (figure printed to file)
    """
    n = len(history.history['val_loss'])

    plt.figure(figsize=(20,10))
    plt.subplot(121)
    plt.plot(np.arange(1,n+1), history.history['dice_coef'], 'b', label='train accuracy')
    plt.plot(np.arange(1,n+1), history.history['val_dice_coef'], 'k', label='validation accuracy')
    plt.xlabel('Epoch number', fontsize=10); plt.ylabel('Dice Coefficient', fontsize=10)
    plt.legend(fontsize=10)

    plt.subplot(122)
    plt.plot(np.arange(1,n+1), history.history['loss'], 'b', label='train loss')
    plt.plot(np.arange(1,n+1), history.history['val_loss'], 'k', label='validation loss')
    plt.xlabel('Epoch number', fontsize=10); plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=10)

    # plt.show()
    plt.savefig(train_hist_fig, dpi=200, bbox_inches='tight')


def make_sample_seg_plot(model, sample_filenames, test_samples_fig, flag='binary'):
    """
    "make_sample_seg_plot"
    This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
    using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
    saving out to the provided filename, cm_filename
    INPUTS:
        * model: trained and compiled keras model
        * sample_filenames: [list] of strings
        * test_samples_fig [string]: filename to print figure to
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (matplotlib figure, printed to file)
    """

    plt.figure(figsize=(16,16))

    for counter,f in enumerate(sample_filenames):
        image = seg_file2tensor(f)/255
        est_label = model.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
        if flag is 'binary':
            est_label[est_label>0.5] = 1
            est_label = (est_label*255).astype(np.uint8)
        else:
            est_label = tf.argmax(est_label, axis=-1)

        if flag is 'binary':
            plt.subplot(6,4,counter+1)
        else:
            plt.subplot(4,4,counter+1)
        name = sample_filenames[counter].split(os.sep)[-1].split('_')[0]
        plt.title(name, fontsize=10)
        plt.imshow(image)
        if flag is 'binary':
            plt.imshow(est_label, alpha=0.5, cmap=plt.cm.gray)
        else:
            plt.imshow(est_label, alpha=0.5, cmap=plt.cm.bwr)

        plt.axis('off')

    # plt.show()
    plt.savefig(test_samples_fig,
                dpi=200, bbox_inches='tight')
    plt.close('all')
