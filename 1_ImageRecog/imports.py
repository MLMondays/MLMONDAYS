

## TAMUCC DATA imports
# from tamucc_imports import *

##UNCOMMENT BELOW TO USE NWPU DATA
from nwpu_imports import *

#see mlmondays blog post:
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

##calcs
import tensorflow as tf #numerical operations on gpu
from tensorflow.keras.applications import MobileNetV2 #mobilenet v2 model, used for feature extraction
from tensorflow.keras.applications import VGG16 #vgg model, used for feature extraction
from tensorflow.keras.applications import Xception #xception model, used for feature extraction
import numpy as np #numerical operations on cpu
from sklearn.decomposition import PCA  #for data dimensionality reduction / viz.
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE #for data dimensionality reduction / viz.

## plots
import matplotlib.pyplot as plt #for plotting
from sklearn.metrics import confusion_matrix #compute confusion matrix from vectors of observed and estimated labels
import seaborn as sns #extended functionality / style to matplotlib plots
from matplotlib.offsetbox import OffsetImage, AnnotationBbox #for visualizing image thumbnails plotted as markers

##i/o
# from matplotlib.image import imread
import pandas as pd #for data wrangling. We just use it to read csv files
import json, shutil #json for class file reading, shutil for file copying/moving

##utils
from sklearn.utils import class_weight #utility for computinh normalised class weights
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K #access to keras backend functions

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


###############################################################
### TFRECORD FUNCTIONS
###############################################################

#-----------------------------------
def resize_and_crop_image(image, label):
    """
    This function crops to square and resizes an image
    The label passes through unmodified
    INPUTS:
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS:
    """
    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = TARGET_SIZE
    th = TARGET_SIZE
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                 )
    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    return image, label

#-----------------------------------
def recompress_image(image, label):
    """
    This function takes an image encoded as a byte string
    and recodes as an 8-bit jpeg
    Label passes through unmodified
    INPUTS:
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS:
    """
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
    return image, label

#-----------------------------------
"""
These functions cast inputs into tf dataset 'feature' classes
There is one for bytestrings (images), one for floats (not used here) and one for ints (labels)
"""
def _bytestring_feature(list_of_bytestrings):
    """
    INPUTS:
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints):
    """
    INPUTS:
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS:
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats):
    """
    INPUTS:
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS:
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

#-----------------------------------
def to_tfrecord(img_bytes, label, CLASSES):
    """
    This function
    INPUTS:
    OPTIONAL INPUTS:
    OUTPUTS:
    GLOBAL INPUTS:
    """
    class_num = np.argmax(np.array(CLASSES)==label)
    feature = {
      "image": _bytestring_feature([img_bytes]), # one image in the list
      "class": _int_feature([class_num]),        # one class in the list
              }
    return tf.train.Example(features=tf.train.Features(feature=feature))

#-----------------------------------
def read_tfrecord(example):
    """
    This function
    INPUTS:
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS:
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])

    class_label = tf.cast(example['class'], tf.int32)

    return image, class_label

#-----------------------------------
def read_image_and_label(img_path):
    """
    This function
    INPUTS:
    OPTIONAL INPUTS:
    OUTPUTS:
    GLOBAL INPUTS:
    """
    bits = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(bits)

    label = tf.strings.split(img_path, sep='/')
    label = tf.strings.split(label[-1], sep='_IMG')

    return image,label[0]

#-----------------------------------
def get_dataset_for_tfrecords(recoded_dir, shared_size):
    """
    This function
    INPUTS:
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS:
    """
    tamucc_dataset = tf.data.Dataset.list_files(recoded_dir+os.sep+'*.jpg', seed=10000) # This also shuffles the images
    tamucc_dataset = tamucc_dataset.map(read_image_and_label)
    tamucc_dataset = tamucc_dataset.map(resize_and_crop_image, num_parallel_calls=AUTO)

    tamucc_dataset = tamucc_dataset.map(recompress_image, num_parallel_calls=AUTO)
    tamucc_dataset = tamucc_dataset.batch(shared_size)
    return tamucc_dataset

#-----------------------------------
def write_records(tamucc_dataset, tfrecord_dir, CLASSES):
    """
    This function
    INPUTS:
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS:
    """
    for shard, (image, label) in enumerate(tamucc_dataset):
      shard_size = image.numpy().shape[0]
      filename = tfrecord_dir+os.sep+"tamucc" + "{:02d}-{}.tfrec".format(shard, shard_size)

      with tf.io.TFRecordWriter(filename) as out_file:
        for i in range(shard_size):
          example = to_tfrecord(image.numpy()[i],label.numpy()[i], CLASSES)
          out_file.write(example.SerializeToString())
        print("Wrote file {} containing {} records".format(filename, shard_size))

###############################################################
### PLOTTING FUNCTIONS
###############################################################

#-----------------------------------
def plot_history(history, train_hist_fig):
    """
    This function
    INPUTS:
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS:
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
    This function
    INPUTS:
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS:
    """
    labs = []
    preds = []
    for img, lab in val_ds.take(-1):
        #print(lab)
        labs.append(lab.numpy().flatten())

        scores = model.predict(img) # tf.expand_dims(img, 0) , batch_size=1)
        n = np.argmax(scores, axis=1)
        preds.append(n)


    labs = np.hstack(labs)
    preds = np.hstack(preds)
    return labs, preds

#-----------------------------------
def p_confmat(labs, preds, cm_filename, CLASSES, thres = 0.1):
    """
    This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
    using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
    saving out to the provided filename, cm_filename
    INPUTS:
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS:
    """
    cm = confusion_matrix(labs, preds)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cm[cm<thres] = 0

    plt.figure(figsize=(15,15))
    sns.heatmap(cm,
        annot=True,
        cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True))

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
    This function computes a confusion matrix (matrix of correspondences between true and estimated classes)
    using the sklearn function of the same name. Then normalizes by column totals, and makes a heatmap plot of the matrix
    saving out to the provided filename, cm_filename
    INPUTS:
        * model
        * sample_filenames
        * test_samples_fig
        * CLASSES
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: matplotlib figure, printed to file
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
    GLOBAL INPUTS: matplotlib figure
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
# Show images with t-SNE
# Source: https://www.kaggle.com/gaborvecsei/plants-t-sne
def visualize_scatter_with_images(X_2d_data, labels, images, figsize=(15,15), image_zoom=1,xlim = (-3,3), ylim=(-3,3)):
    """
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


###############################################################
### DATASET FUNCTIONS
###############################################################
#-----------------------------------
def read_classes_from_json(json_file):
    """
    This function reads the contents of a json file enumerating classes
    INPUTS:
        * json_file [string]: full path to the json file
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * CLASSES [list]: list of classesd as byte strings
    """
    with open(json_file) as f:
        class_dict = json.load(f)

    # string names
    CLASSES = [class_dict[k] for k in class_dict.keys()]
    #bytestrings names
    CLASSES = [c.encode() for c in CLASSES]
    return CLASSES

#-----------------------------------
def file2tensor(f, model='mobilenet'):
    """
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained mobilenet or vgg model
    (the imagery is standardized depedning on target model framework)
    INPUTS:
    OPTIONAL INPUTS:
    OUTPUTS:
    GLOBAL INPUTS:
    """
    bits = tf.io.read_file(f)
    image = tf.image.decode_jpeg(bits)

    # image = tf.image.resize(image, (TARGET_SIZE, TARGET_SIZE))

    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = TARGET_SIZE
    th = TARGET_SIZE
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                  lambda: tf.image.resize(image, [w*tw/w, h*tw/w]), # if true
                  lambda: tf.image.resize(image, [w*th/h, h*th/h])  # if false
                 )
    image = tf.keras.preprocessing.image.load_img(f, target_size=(TARGET_SIZE, TARGET_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)

    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    # image = tf.cast(image, tf.uint8) #/ 255.0
    if model == 'mobilenet':
       im = tf.keras.applications.mobilenet_v2.preprocess_input(image) #specific to mobilenetV2
    elif model=='vgg':
       im = tf.keras.applications.vgg16.preprocess_input(image) #specific to vgg16

    return image, im

#-----------------------------------
def get_batched_dataset(filenames):
    """
    This function defines a workflow for the model to read data from
    tfrecord files by defining the degree of parallelism, batch size, pre-fetching, etc
    and also formats the imagery properly for model training
    (assumes mobilenet by using read_tfrecord_mv2)
    INPUTS:
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS:
    """
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = True # False

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    dataset = dataset.map(read_tfrecord_mv2, num_parallel_calls=AUTO)

    dataset = dataset.cache() # This dataset fits in RAM
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(AUTO) #

    return dataset

#-----------------------------------
def get_eval_dataset(filenames):
    """
    This function defines a workflow for the model to read data from
    tfrecord files by defining the degree of parallelism, batch size, pre-fetching, etc
    and also formats the imagery properly for model training
    (assumes mobilenet by using read_tfrecord_mv2)

    This evaluation version does not .repeat() because it is not being called repeatedly by a model
    INPUTS:
    OPTIONAL INPUTS:
    GLOBAL INPUTS:
    OUTPUTS:
    """
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = True #False

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=BATCH_SIZE, num_parallel_calls=AUTO)
    dataset = dataset.map(read_tfrecord_mv2, num_parallel_calls=AUTO)

    dataset = dataset.cache() # This dataset fits in RAM
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(AUTO) #

    return dataset


#-----------------------------------
def read_tfrecord_vgg(example):
    """
    This function reads an example record from a tfrecord file
    and parses into label and image ready for vgg model training
    INPUTS:
        * example: an tfrecord 'example' object, containing an image and label
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor]: resized and pre-processed for vgg
        * class_label [tensor] 32-bit integer
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) #/ 255.0

    image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])
    #image = tf.image.per_image_standardization(image)
    image = tf.keras.applications.vgg.preprocess_input(image) #specific to model

    class_label = tf.cast(example['class'], tf.int32)

    return image, class_label


#-----------------------------------
def read_tfrecord_mv2(example):
    """
    This function reads an example record from a tfrecord file
    and parses into label and image ready for mobilenet model training
    INPUTS:
        * example: an tfrecord 'example' object, containing an image and label
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS:
        * image [tensor]: resized and pre-processed for mobilenetv2
        * class_label [tensor] 32-bit integer
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) #/ 255.0

    image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])
    #image = tf.image.per_image_standardization(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image) #specific to model

    class_label = tf.cast(example['class'], tf.int32)

    return image, class_label

###############################################
##### MODEL FUNCTIONS
###############################################
# learning rate function
def lrfn(epoch):
    """
    This function creates a custom piecewise linear-exponential learning rate function
    for a custom learning rate scheduler. It is linear to a max, then exponentially decays
    INPUTS: current epoch number
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay
    OUTPUTS:  the function lr with all arguments passed
    """
    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        if epoch < rampup_epochs:
            lr = (max_lr - start_lr)/rampup_epochs * epoch + start_lr
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        else:
            lr = (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr
        return lr
    return lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)


#-----------------------------------
def transfer_learning_model_vgg(num_classes, input_shape, dropout_rate=0.5):
    """
    This function creates an implementation of a convolutional deep learning model for estimating
	a discrete category based on vgg, trained using transfer learning
    (initialized using pretrained imagenet weights)
    INPUTS:
        * num_classes = number of classes (output nodes on classification layer)
        * input_shape = size of input layer (i.e. image tensor)
    OPTIONAL INPUTS:
        * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
    GLOBAL INPUTS: None
    OUTPUTS: keras model instance
    """
    EXTRACTOR = VGG16(weights="imagenet", include_top=False,
                        input_shape=input_shape)

    EXTRACTOR.trainable = False
    # Construct the head of the model that will be placed on top of the
    # the base model
    class_head = EXTRACTOR.output
    class_head = tf.keras.layers.GlobalAveragePooling2D()(class_head)
    class_head = tf.keras.layers.Dense(256, activation="relu")(class_head)
    class_head = tf.keras.layers.Dropout(dropout_rate)(class_head)
    class_head = tf.keras.layers.Dense(num_classes, activation="softmax")(class_head)

    # Create the new model
    model = tf.keras.Model(inputs=EXTRACTOR.input, outputs=class_head)

    return model

#-----------------------------------
def mobilenet_model(num_classes, input_shape, dropout_rate=0.5):
    """
    This function creates an implementation of a convolutional deep learning model for estimating
	a discrete category based on mobilenet, trained from scratch
    INPUTS:
        * num_classes = number of classes (output nodes on classification layer)
        * input_shape = size of input layer (i.e. image tensor)
    OPTIONAL INPUTS:
        * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
    GLOBAL INPUTS: None
    OUTPUTS: keras model instance
    """
    EXTRACTOR = MobileNetV2(weights=None, include_top=False,
                        input_shape=input_shape)

    EXTRACTOR.trainable = True
    # Construct the head of the model that will be placed on top of the
    # the base model
    class_head = EXTRACTOR.output
    class_head = tf.keras.layers.GlobalAveragePooling2D()(class_head)
    class_head = tf.keras.layers.Dense(256, activation="relu")(class_head)
    class_head = tf.keras.layers.Dropout(dropout_rate)(class_head)
    class_head = tf.keras.layers.Dense(num_classes, activation="softmax")(class_head)

    # Create the new model
    model = tf.keras.Model(inputs=EXTRACTOR.input, outputs=class_head)

    return model

#-----------------------------------
def transfer_learning_mobilenet_model(num_classes, input_shape, dropout_rate=0.5):
    """
    This function creates an implementation of a convolutional deep learning model for estimating
	a discrete category based on mobilenet v2, trained using transfer learning
    (initialized using pretrained imagenet weights)
    INPUTS:
        * num_classes = number of classes (output nodes on classification layer)
        * input_shape = size of input layer (i.e. image tensor)
    OPTIONAL INPUTS:
        * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
    GLOBAL INPUTS: None
    OUTPUTS: keras model instance
    """
    EXTRACTOR = MobileNetV2(weights="imagenet", include_top=False,
                        input_shape=input_shape)

    EXTRACTOR.trainable = False
    # Construct the head of the model that will be placed on top of the
    # the base model
    class_head = EXTRACTOR.output
    class_head = tf.keras.layers.GlobalAveragePooling2D()(class_head)
    class_head = tf.keras.layers.Dense(256, activation="relu")(class_head)
    class_head = tf.keras.layers.Dropout(dropout_rate)(class_head)
    class_head = tf.keras.layers.Dense(num_classes, activation="softmax")(class_head)

    # Create the new model
    model = tf.keras.Model(inputs=EXTRACTOR.input, outputs=class_head)

    return model

#-----------------------------------
def transfer_learning_xception_model(num_classes, input_shape, dropout_rate=0.25):
    """
    This function creates an implementation of a convolutional deep learning model for estimating
	a discrete category based on xception, trained using transfer learning
    (initialized using pretrained imagenet weights)
    INPUTS:
        * num_classes = number of classes (output nodes on classification layer)
        * input_shape = size of input layer (i.e. image tensor)
    OPTIONAL INPUTS:
        * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
    GLOBAL INPUTS: None
    OUTPUTS: keras model instance
    """
    EXTRACTOR = Xception(weights="imagenet", include_top=False,
                        input_shape=input_shape)

    EXTRACTOR.trainable = False
    # Construct the head of the model that will be placed on top of the
    # the base model
    class_head = EXTRACTOR.output
    class_head = tf.keras.layers.GlobalAveragePooling2D()(class_head)
    class_head = tf.keras.layers.Dense(256, activation="relu")(class_head)
    class_head = tf.keras.layers.Dropout(dropout_rate)(class_head)
    class_head = tf.keras.layers.Dense(num_classes, activation="softmax")(class_head)

    # Create the new model
    model = tf.keras.Model(inputs=EXTRACTOR.input, outputs=class_head)

    return model

#-----------------------------------
def xception_model(num_classes, input_shape, dropout_rate=0.25):
    """
    This function creates an implementation of a convolutional deep learning model for estimating
	a discrete category based on xception, trained from scratch
    INPUTS:
        * num_classes = number of classes (output nodes on classification layer)
        * input_shape = size of input layer (i.e. image tensor)
    OPTIONAL INPUTS:
        * dropout_rate = proportion of neurons to randomly set to zero, after the pooling layer
    GLOBAL INPUTS: None
    OUTPUTS: keras model instance
    """
    EXTRACTOR = Xception(weights=None, include_top=False,
                        input_shape=input_shape)

    EXTRACTOR.trainable = True
    # Construct the head of the model that will be placed on top of the
    # the base model
    class_head = EXTRACTOR.output
    class_head = tf.keras.layers.GlobalAveragePooling2D()(class_head)
    class_head = tf.keras.layers.Dense(256, activation="relu")(class_head)
    class_head = tf.keras.layers.Dropout(dropout_rate)(class_head)
    class_head = tf.keras.layers.Dense(num_classes, activation="softmax")(class_head)

    # Create the new model
    model = tf.keras.Model(inputs=EXTRACTOR.input, outputs=class_head)

    return model

###===================================================
def conv_block(inp, filters=32, bn=True, pool=True):
    """
    This function generates a convolutional block
    INPUTS:
        * inp = input layer
    OPTIONAL INPUTS:
        * filters = number of convolutional filters to use
        * bn=False, use batch normalization in each convolutional layer
        * pool=True, use pooling in each convolutional layer
        * shallow=True, if False, a larger model with more convolution layers is used
    GLOBAL INPUTS: None
    OUTPUTS: keras model layer object
    """
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, activation='relu',
              kernel_initializer='he_uniform')(inp)

    # _ = SeparableConv2D(filters=filters, kernel_size=3, activation='relu')(inp) #kernel_initializer='he_uniform'
    if bn:
       x = tf.keras.layers.BatchNormalization()(x)
    if pool:
       x = tf.keras.layers.MaxPool2D()(x)
    # if drop:
    #     _ = tf.keras.layers.Dropout(0.2)(_)
    return x

###===================================================
def make_cat_model(num_classes, dropout, denseunits, base_filters, bn=False, pool=True, shallow=True):
    """
    This function creates an implementation of a convolutional deep learning model for estimating
	a discrete category
    INPUTS:
        * num_classes = number of classes (output nodes on classification layer)
        * dropout = proportion of neurons to randomly set to zero, after the pooling layer
        * denseunits = number of neurons in the classifying layer
        * base_filters = number of convolutional filters to use in the first layer
    OPTIONAL INPUTS:
        * bn=False, use batch normalization in each convolutional layer
        * pool=True, use pooling in each convolutional layer
        * shallow=True, if False, a larger model with more convolution layers is used
    GLOBAL INPUTS: TARGET_SIZE
    OUTPUTS: keras model instance
    """
    input_layer = tf.keras.layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))

    x = conv_block(input_layer, filters=base_filters, bn=bn, pool=pool)
    x = conv_block(x, filters=base_filters*2, bn=bn, pool=pool)
    x = conv_block(x, filters=base_filters*3, bn=bn, pool=pool)
    x = conv_block(x, filters=base_filters*4, bn=bn, pool=pool)

    if shallow is False:
        x = conv_block(x, filters=base_filters*5, bn=bn, pool=pool)
        x = conv_block(x, filters=base_filters*6, bn=bn, pool=pool)

    bottleneck = tf.keras.layers.GlobalMaxPool2D()(x)
    bottleneck = tf.keras.layers.Dropout(dropout)(bottleneck)

    # for class prediction
    class_head = tf.keras.layers.Dense(units=denseunits, activation='relu')(bottleneck)
    class_head = tf.keras.layers.Dense(units=num_classes, activation='softmax', name='output')(class_head)

    model = tf.keras.models.Model(inputs=input_layer, outputs=[class_head])

    model.compile(optimizer='adam',
              loss={'output': 'categorical_crossentropy'},
              metrics={'output': 'accuracy'})

    return model
