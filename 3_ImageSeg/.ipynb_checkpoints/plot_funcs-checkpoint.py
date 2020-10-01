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

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels


###############################################################
### PLOTTING FUNCTIONS
###############################################################

#-----------------------------------
def crf_refine(label, img):
    """
    "crf_refine(label, img)"
    This function refines a label image based on an input label image and the associated image
    Uses a conditional random field algorithm using spatial and image features
    INPUTS:
        * label [ndarray]: label image 2D matrix of integers
        * image [ndarray]: image 3D matrix of integers
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: label [ndarray]: label image 2D matrix of integers
    """
    H = label.shape[0]
    W = label.shape[1]
    U = unary_from_labels(1+label,5,gt_prob=0.51)
    d = dcrf.DenseCRF2D(H, W, 5)
    d.setUnaryEnergy(U)

    # to add the color-independent term, where features are the locations only:
    d.addPairwiseGaussian(sxy=(3, 3),
                 compat=3,
                 kernel=dcrf.DIAG_KERNEL,
                 normalization=dcrf.NORMALIZE_SYMMETRIC)
    feats = create_pairwise_bilateral(
                          sdims=(100, 100),
                          schan=(2,2,2),
                          img=img,
                          chdim=2)

    d.addPairwiseEnergy(feats, compat=120,kernel=dcrf.DIAG_KERNEL,normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(10)
    kl1 = d.klDivergence(Q)
    return np.argmax(Q, axis=0).reshape((H, W)).astype(np.uint8), kl1


#-----------------------------------
def plot_seg_history_iou(history, train_hist_fig):
    """
    "plot_seg_history_iou(history, train_hist_fig)"
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
    "plot_seg_history(history, train_hist_fig)"
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
    "make_sample_seg_plot(model, sample_filenames, test_samples_fig, flag='binary')"
    This function uses a trained model to estimate the label image from each input image
    and returns both images and labels as a list
    INPUTS:
        * model: trained and compiled keras model
        * sample_filenames: [list] of strings
        * test_samples_fig [string]: filename to print figure to
        * flag [string]: either 'binary' or 'multiclass'
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * imgs: [list] of images
        * lbls: [list] of label images
    """

    plt.figure(figsize=(16,16))
    imgs = []
    lbls = []

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
            plt.imshow(est_label, alpha=0.5, cmap=plt.cm.gray, vmin=0, vmax=1)
        else:
            plt.imshow(est_label, alpha=0.5, cmap=plt.cm.bwr, vmin=0, vmax=3)

        plt.axis('off')
        imgs.append((image.numpy()*255).astype(np.uint8))
        lbls.append(est_label)

    # plt.show()
    plt.savefig(test_samples_fig,
                dpi=200, bbox_inches='tight')
    plt.close('all')

    return imgs, lbls


###---------------------------------------------------------
def make_sample_ensemble_seg_plot(model2, model3, sample_filenames, test_samples_fig, flag='binary'):
    """
    "make_sample_ensemble_seg_plot(model2, model3, sample_filenames, test_samples_fig, flag='binary')"
    This function uses two trained models to estimate the label image from each input image
    It then uses a KL score to determine which one to return
    and returns both images and labels as a list, as well as a list of which model's output is returned
    INPUTS:
        * model: trained and compiled keras model
        * sample_filenames: [list] of strings
        * test_samples_fig [string]: filename to print figure to
        * flag [string]: either 'binary' or 'multiclass'
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * imgs: [list] of images
        * lbls: [list] of label images
        * model_num: [list] of integers indicating which model's output was retuned based on CRF KL divergence
    """

    plt.figure(figsize=(16,16))
    imgs = []
    lbls = []
    model_num = []

    for counter,f in enumerate(sample_filenames):
        image = seg_file2tensor(f)/255
        est_label1 = model2.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
        if flag is 'binary':
            est_label1[est_label1>0.5] = 1
            est_label1 = (est_label1*255).astype(np.uint8)
        else:
            est_label1 = tf.argmax(est_label1, axis=-1)


        est_label2 = model3.predict(tf.expand_dims(image, 0) , batch_size=1).squeeze()
        if flag is 'binary':
            est_label2[est_label2>0.5] = 1
            est_label2 = (est_label2*255).astype(np.uint8)
        else:
            est_label2 = tf.argmax(est_label2, axis=-1)

        label = est_label1.numpy().astype('int')
        img = (image.numpy()*255).astype(np.uint8)
        est_labelA, kl1 = crf_refine(label, img )
        label = est_label2.numpy().astype('int')
        est_labelB, kl2 = crf_refine(label, img )
        del label

        # plt.subplot(221); plt.imshow(image); plt.imshow(est_label1, alpha=0.5, cmap=plt.cm.bwr, vmin=0, vmax=3); plt.axis('off'); plt.title('Model 1 estimate', fontsize=6)
        # plt.subplot(222); plt.imshow(image); plt.imshow(est_label2, alpha=0.5, cmap=plt.cm.bwr, vmin=0, vmax=3); plt.axis('off'); plt.title('Model 2 estimate', fontsize=6)
        # plt.subplot(223); plt.imshow(image); plt.imshow(est_labelA, alpha=0.5, cmap=plt.cm.bwr, vmin=0, vmax=3); plt.axis('off'); plt.title('Model 1 CRF estimate ('+str(-np.log(-kl1))[:7]+')', fontsize=6)
        # plt.subplot(224); plt.imshow(image); plt.imshow(est_labelB, alpha=0.5, cmap=plt.cm.bwr, vmin=0, vmax=3); plt.axis('off'); plt.title('Model 2 CRF estimate ('+str(-np.log(-kl2))[:7]+')', fontsize=6)
        # plt.savefig('crf-example'+str(counter)+'.png', dpi=600, bbox_inches='tight'); plt.close('all')
        #

        if kl1<kl2:
            est_label = est_labelA.copy()
            model_num.append(1)
        else:
            est_label = est_labelB.copy()
            model_num.append(2)

        if flag is 'binary':
            plt.subplot(6,4,counter+1)
        else:
            plt.subplot(4,4,counter+1)
        name = sample_filenames[counter].split(os.sep)[-1].split('_')[0]
        plt.title(name, fontsize=10)
        plt.imshow(image)
        if flag is 'binary':
            plt.imshow(est_label, alpha=0.5, cmap=plt.cm.gray, vmin=0, vmax=1)
        else:
            plt.imshow(est_label, alpha=0.5, cmap=plt.cm.bwr, vmin=0, vmax=3)

        plt.axis('off')

        imgs.append(image)
        lbls.append(est_label)

    # plt.show()
    plt.savefig(test_samples_fig,
                dpi=200, bbox_inches='tight')
    plt.close('all')

    return imgs, lbls, model_num
