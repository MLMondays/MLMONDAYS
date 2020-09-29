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

#see mlmondays blog post:
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np #numerical operations on cpu

# set a seed for reproducibility
SEED=42
np.random.seed(SEED)

## plots
import matplotlib.pyplot as plt #for plotting
from sklearn.metrics import confusion_matrix #compute confusion matrix from vectors of observed and estimated labels
# import seaborn as sns #extended functionality / style to matplotlib plots
from matplotlib.offsetbox import OffsetImage, AnnotationBbox #for visualizing image thumbnails plotted as markers

#calcs
from sklearn.decomposition import PCA  #for data dimensionality reduction / viz.
from sklearn.preprocessing import StandardScaler #data scaling data in PCA and TSNE algorithms
from sklearn.manifold import TSNE #for data dimensionality reduction / viz.

from tfrecords_funcs import file2tensor
import tensorflow as tf #numerical operations on gpu


###############################################################
### PLOTTING FUNCTIONS
###############################################################

#-----------------------------------
def plot_history(history, train_hist_fig):
    """
    plot_history(history, train_hist_fig)
    This function plots the training history of a model
    INPUTS:
        * history [dict]: the output dictionary of the model.fit() process, i.e. history = model.fit(...)
        * train_hist_fig [string]: the filename where the plot will be printed
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (figure printed to file)
    """
    n = len(history.history['accuracy'])

    plt.figure(figsize=(20,10))
    plt.subplot(121)
    plt.plot(np.arange(1,n+1), history.history['accuracy'], 'b', label='train accuracy')
    plt.plot(np.arange(1,n+1), history.history['val_accuracy'], 'k', label='validation accuracy')
    plt.xlabel('Epoch number', fontsize=10); plt.ylabel('Accuracy', fontsize=10)
    plt.legend(fontsize=10)

    plt.subplot(122)
    plt.plot(np.arange(1,n+1), history.history['loss'], 'b', label='train loss')
    plt.plot(np.arange(1,n+1), history.history['val_loss'], 'k', label='validation loss')
    plt.xlabel('Epoch number', fontsize=10); plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=10)

    # plt.show()
    plt.savefig(train_hist_fig, dpi=200, bbox_inches='tight')

#-----------------------------------
def get_label_pairs(val_ds, model):
    """
    get_label_pairs(val_ds, model)
    This function gets label observations and model estimates
    INPUTS:
        * val_ds: a batched data set object
        * model: trained and compiled keras model instance
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * labs [ndarray]: 1d vector of numeric labels
        * preds [ndarray]: 1d vector of correspodning model predicted numeric labels
    """
    labs = []
    preds = []
    for img, lab in val_ds.take(-1):
        labs.append(lab.numpy().flatten())
        scores = model.predict(img)
        n = np.argmax(scores, axis=1)
        preds.append(n)

    labs = np.hstack(labs)
    preds = np.hstack(preds)
    return labs, preds

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
    plt.xticks(tick_marks, [c.decode() for c in CLASSES], rotation=45,fontsize=12)
    plt.yticks(tick_marks, [c.decode() for c in CLASSES],rotation=45, fontsize=12)

    plt.savefig(cm_filename,
                dpi=200, bbox_inches='tight')
    plt.close('all')

    print("Average true positive rate across %i classes: %.3f" % (len(CLASSES), np.mean(np.diag(cm))))


###############################################################
### DATA VIZ FUNCTIONS
###############################################################

def make_sample_plot(model, sample_filenames, test_samples_fig, CLASSES):
    """
    make_sample_plot(model, sample_filenames, test_samples_fig, CLASSES))
    This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
    using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
    saving out to the provided filename, cm_filename
    INPUTS:
        * model: trained and compiled keras model
        * sample_filenames: [list] of strings
        * test_samples_fig [string]: filename to print figure to
        * CLASSES [list] os trings: class names
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (matplotlib figure, printed to file)
    """

    plt.figure(figsize=(16,16))

    for counter,f in enumerate(sample_filenames):
        image, im = file2tensor(f, 'mobilenet')
        plt.subplot(6,4,counter+1)
        name = sample_filenames[counter].split(os.sep)[-1].split('_')[0]
        plt.title(name, fontsize=10)
        plt.imshow(tf.cast(image, tf.uint8))
        plt.axis('off')

        scores = model.predict(tf.expand_dims(im, 0) , batch_size=1)
        n = np.argmax(scores[0])
        est_name = CLASSES[n].decode()
        if name==est_name:
           plt.text(10,50,'prediction: %s' % est_name,
                    color='k', fontsize=12,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round",
                           ec=(.1, 1., .5),
                           fc=(.1, 1., .5),
                           ))
        else:
           plt.text(10,50,'prediction: %s' % est_name,
                    color='k', fontsize=12,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round",
                           ec=(1., 0.5, 0.1),
                           fc=(1., 0.8, 0.8),
                           ))

    # plt.show()
    plt.savefig(test_samples_fig,
                dpi=200, bbox_inches='tight')
    plt.close('all')


#-----------------------------------
def compute_hist(images):
    """
    compute_hist(images)
    Compute the per channel histogram for a batch
    of images
    INPUTS:
        * images [ndarray]: batch of shape (N x W x H x 3)
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * hist_r [dict]: histogram frequencies {'hist'} and bins {'bins'} for red channel
        * hist_g [dict]: histogram frequencies {'hist'} and bins {'bins'} for green channel
        * hist_b [dict]: histogram frequencies {'hist'} and bins {'bins'} for blue channel
    """
    images = images/255.
    mean = np.mean(images, axis=0, dtype=np.float64)

    mean_r, mean_g, mean_b = mean[:,:,0], mean[:,:,1], mean[:,:,2]
    mean_r = np.reshape(mean_r, (-1, 1))
    mean_g = np.reshape(mean_g, (-1, 1))
    mean_b = np.reshape(mean_b, (-1, 1))

    hist_r_, bins_r = np.histogram(mean_r, bins="auto")
    hist_g_, bins_g = np.histogram(mean_g, bins="auto")
    hist_b_, bins_b = np.histogram(mean_b, bins="auto")

    hist_r = {"hist": hist_r_, "bins": bins_r}
    hist_g = {"hist": hist_g_, "bins": bins_g}
    hist_b = {"hist": hist_b_, "bins": bins_b}

    return hist_r, hist_g, hist_b

#-----------------------------------
def plot_distribution(images, labels, class_id, CLASSES):
    """
    plot_distribution(images, labels, class_id, CLASSES)
    Compute the per channel histogram for a batch
    of images
    INPUTS:
        * images [ndarray]: batch of shape (N x W x H x 3)
        * labels [ndarray]: batch of shape (N x 1)
        * class_id [int]: class integer to plot
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: matplotlib figure
    """
    fig = plt.figure(figsize=(21,7))
    rows, cols = 1, 3
    locs = np.where(labels == class_id)
    samples = locs[:][0]
    class_images = images[samples]
    hist_r, hist_g, hist_b = compute_hist(class_images)
    plt.title("Histogram - Mean Pixel Value:  " + CLASSES[class_id])
    plt.axis('off')

    fig.add_subplot(rows, cols, 1)
    hist, bins = hist_r["hist"], hist_r["bins"]
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width,color='r')
    plt.xlim((0,1))
    plt.ylim((0, 255))

    fig.add_subplot(rows, cols, 2)
    hist, bins = hist_g["hist"], hist_g["bins"]
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width,color='g')
    plt.xlim((0,1))
    plt.ylim((0,255))

    fig.add_subplot(rows, cols, 3)
    hist, bins = hist_b["hist"], hist_b["bins"]
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width,color='b')
    plt.xlim((0,1))
    plt.ylim((0, 255))

#-----------------------------------
def plot_one_class(inp_batch, sample_idx, label, batch_size, CLASSES, rows=8, cols=8, size=(20,15)):
    """
    plot_one_class(inp_batch, sample_idx, label, batch_size, CLASSES, rows=8, cols=8, size=(20,15)):
    Plot "batch_size" images that belong to the class "label"
    INPUTS:
        * inp_batch
        * sample_idx
        * label
        * batch_size
    OPTIONAL INPUTS:
        * rows=8
        * cols=8
        * size=(20,15)
    GLOBAL INPUTS: None (matplotlib figure, printed to file)
    """

    fig = plt.figure(figsize=size)
    plt.title(CLASSES[int(label)])
    plt.axis('off')
    for n in range(0, batch_size):
        fig.add_subplot(rows, cols, n + 1)
        img = inp_batch[n]
        plt.imshow(img)
        plt.axis('off')

#-----------------------------------
def compute_mean_image(images, opt="mean"):
    """
    compute_mean_image(images, opt="mean")
    Compute and return mean image given
    a batch of images
    INPUTS:
        * images [ndarray]: batch of shape (N x W x H x 3)
    OPTIONAL INPUTS:
        * opt="mean" or "median"
    GLOBAL INPUTS:
    OUTPUTS: 2d mean image [ndarray]
    """
    images = images/255.
    if opt == "mean":
        return np.mean(images, axis=0, dtype=np.float64)
    else:
        return np.median(images, axis=0)

#-----------------------------------
def plot_mean_images(images, labels, CLASSES, rows=3, cols = 2):
    """
    plot_mean_images(images, labels, CLASSES, rows=3, cols = 2)
    Plot the mean image of a set of images
    INPUTS:
        * images [ndarray]: batch of shape (N x W x H x 3)
        * labels [ndarray]: batch of shape (N x 1)
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS: matplotlib figure
    """
    fig = plt.figure(figsize=(20,15))

    example_images = []
    for n in np.arange(len(CLASSES)):
        fig.add_subplot(rows, cols, n + 1)
        locs = np.where(labels == n)
        samples = locs[:][0]
        class_images = images[samples]
        img = compute_mean_image(class_images, "median")
        plt.imshow(img)
        plt.title(CLASSES[n])
        plt.axis('off')

#-----------------------------------
def plot_tsne(tsne_result, label_ids, CLASSES):
    """
    plot_tsne(tsne_result, label_ids, CLASSES)
    Plot TSNE loadings and colour code by class
    Source: https://www.kaggle.com/gaborvecsei/plants-t-sne
    INPUTS:
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS: matplotlib figure, matplotlib figure axes object
    """
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111,projection='3d')

    plt.grid()

    nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):
      ax.scatter(tsne_result[np.where(label_ids == label_id), 0],
                  tsne_result[np.where(label_ids == label_id), 1],
                  tsne_result[np.where(label_ids == label_id), 2],
                  alpha=0.8,
                  color= plt.cm.Set1(label_id / float(nb_classes)),
                  marker='o',
                  label=CLASSES[label_id])
    ax.legend(loc='best')
    ax.axis('tight')

    ax.view_init(25, 45)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-2.5, 2.5)
    return fig, ax

#-----------------------------------
def visualize_scatter_with_images(X_2d_data, labels, images, figsize=(15,15), image_zoom=1,xlim = (-3,3), ylim=(-3,3)):
    """
    visualize_scatter_with_images(X_2d_data, labels, images, figsize=(15,15), image_zoom=1,xlim = (-3,3), ylim=(-3,3))
    Plot TSNE loadings and colour code by class
    Source: https://www.kaggle.com/gaborvecsei/plants-t-sne
    INPUTS:
        * X_2d_data
        * images
    OPTIONAL INPUTS:
        * figsize=(15,15)
        * image_zoom=1
    GLOBAL INPUTS:
    OUTPUTS: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0, _ = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data[:,:2])
    ax.autoscale()
    ax.axis('tight')
    ax.scatter(X_2d_data[:,0], X_2d_data[:,1], 20, labels, zorder=10)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return fig
