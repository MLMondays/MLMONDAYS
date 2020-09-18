
TARGET_SIZE = 400
VALIDATION_SPLIT = 0.5 #0.6
ims_per_shard = 200
BATCH_SIZE = 6

num_classes = 12 #12 # 4 #2

#see mlmondays blog post:
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

##calcs
import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


##plots
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import seaborn as sns


##utils
from collections import defaultdict
from PIL import Image
from collections import Counter
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import json, shutil
import pandas as pd

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import tensorflow.keras.backend as K

###############################################################
### MODEL FUNCTIONS
###############################################################

# code repurposed from https://keras.io/examples/vision/metric_learning/

class EmbeddingModel(tf.keras.Model):
    """
    This function
    """
    def train_step(self, data):
        # Note: Workaround for open issue, to be removed.
        if isinstance(data, tuple):
            data = data[0]
        anchors, positives = data[0], data[1]

        with tf.GradientTape() as tape:
            # Run both anchors and positives through model.
            anchor_embeddings = self(anchors, training=True)
            positive_embeddings = self(positives, training=True)

            # Calculate cosine similarity between anchors and positives. As they have
            # been normalised this is just the pair wise dot products.
            similarities = tf.einsum(
                "ae,pe->ap", anchor_embeddings, positive_embeddings
            )

            # Since we intend to use these as logits we scale them by a temperature.
            # This value would normally be chosen as a hyper parameter.
            temperature = 0.3 # 0.2
            similarities /= temperature

            # We use these similarities as logits for a softmax. The labels for
            # this call are just the sequence [0, 1, 2, ..., num_classes] since we
            # want the main diagonal values, which correspond to the anchor/positive
            # pairs, to be high. This loss will move embeddings for the
            # anchor/positive pairs together and move all other pairs apart.
            sparse_labels = tf.range(num_classes)
            # sparse_labels[0] = sparse_labels[0]*class_weights[0]
            # sparse_labels[1] = sparse_labels[1]*class_weights[1]
            loss = self.compiled_loss(sparse_labels, similarities)

        # Calculate gradients and apply via optimizer.
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (specifically the one for the loss value).
        self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}


# def get_gram_matrix(X_test, model):
#     embeddings = model.predict(X_test)
#
#     embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
#
#     gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
#     return gram_matrix


# def near_neighbours_from_samples(X_train, model, near_neighbours_per_example):
#     embeddings = model.predict(X_train)
#
#     embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
#
#     gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
#     near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]
#     return near_neighbours


# def get_cm_from_near_neighbours(ytrain, num_classes, near_neighbours, near_neighbours_per_example, class_idx_to_idxs):
#
#     confusion_matrix = np.zeros((num_classes, num_classes))
#
#     # For each class.
#     for class_idx in range(num_classes):
#         # Consider "near_neighbours_per_example" examples.
#         example_idxs = class_idx_to_idxs[class_idx][:near_neighbours_per_example]
#         for y_train_idx in example_idxs:
#             # And count the classes of its near neighbours.
#             for nn_idx in near_neighbours[y_train_idx][:-1]:
#                 nn_class_idx = ytrain[nn_idx]
#                 confusion_matrix[class_idx, nn_class_idx] += 1
#     confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
#     return confusion_matrix


def get_large_embedding_model(TARGET_SIZE, num_classes, num_embed_dim):
    """
    This function
    """
    inputs = tf.keras.layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, activation="relu")(inputs) #
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu")(inputs) #
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu")(x) #32
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu")(x) #64
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, activation="relu")(x) #64
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    embeddings = tf.keras.layers.Dense(units = num_embed_dim, activation=None)(x)
    #embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    model = EmbeddingModel(inputs, embeddings)
    return model

def get_embedding_model(TARGET_SIZE, num_classes, num_embed_dim):
    """
    This function
    """
    inputs = tf.keras.layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu")(inputs) #
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu")(x) #32
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu")(x) #64
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    embeddings = tf.keras.layers.Dense(units = num_embed_dim, activation=None)(x)
    #embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    model = EmbeddingModel(inputs, embeddings)
    return model

def fit_knn_to_embeddings(model, X_train, ytrain, num_dim_use):
    """
    This function
    """
    embeddings = model.predict(X_train)
    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(embeddings.numpy()[:,:num_dim_use], ytrain)
    return knn


###############################################################
### PLOT FUNCTIONS
###############################################################

# def conf_matrix(y_test, gram_matrix, max_num_near_neighbours, min_num_near_neighbours, num_classes, class_idx_to_idxs):
#     # code adapted from https://keras.io/examples/vision/metric_learning/
#
#     MN = []; MJ = []
#     for near_neighbours_per_example in np.arange(min_num_near_neighbours,max_num_near_neighbours,1):
#
#       near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]
#
#       confusion_matrix = np.zeros((num_classes, num_classes))
#
#       # For each class.
#       for class_idx in range(num_classes):
#           # Consider 'near_neighbours' examples.
#           example_idxs = class_idx_to_idxs[class_idx][:near_neighbours_per_example]
#           for y_test_idx in example_idxs:
#               # And count the classes of its near neighbours.
#               for nn_idx in near_neighbours[y_test_idx][:-1]:
#                   nn_class_idx = y_test[nn_idx]
#                   #tally that class pairing
#                   confusion_matrix[class_idx, nn_class_idx] += 1
#
#       # normalize by row totals to make the matrix stochastic
#       confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
#
#       # mean recall as the mean of diagonal elements
#       MN.append(np.mean(np.diag(confusion_matrix)))
#       # maximum recall
#       #MJ.append(np.max(np.diag(confusion_matrix)))
#     return MN, confusion_matrix


#-----------------------------------
def conf_mat_filesamples(model, knn, sample_filenames, num_classes, num_dim_use, CLASSES):
    """
    This function
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
    This function
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

    # plt.show()
    plt.savefig(cm_filename,
                dpi=200, bbox_inches='tight')
    plt.close('all')

    print("Average true positive rate across %i classes: %.3f" % (len(CLASSES), np.mean(np.diag(cm))))


###############################################################
### DATA FUNCTIONS
###############################################################

#-----------------------------------
def read_tfrecord(example):
    """
    This function
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.uint8) #float32) / 255.0
    image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])

    class_label = tf.cast(example['class'], tf.int32)

    return image, class_label


## test using imgae read from file
#-----------------------------------
def file2tensor(f):
    """
    This function reads a jpeg image from file into a cropped and resized tensor,
    for use in prediction with a trained model
    """
    bits = tf.io.read_file(f)
    image = tf.image.decode_jpeg(bits)

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
    image = tf.cast(image, tf.uint8) #/ 255.0

    return image

###############################################################
### TFRECORD FUNCTIONS
###############################################################

#-----------------------------------
def recompress_image(image, label):
    """
    This function
    """
    image = tf.cast(image, tf.uint8)
    image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
    return image, label

#-----------------------------------
"""
This function
"""
def _bytestring_feature(list_of_bytestrings):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): # int64
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats): # float32
  return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))

#-----------------------------------
def to_tfrecord(img_bytes, label, CLASSES):
    """
    This function
    """
    class_num = np.argmax(np.array(CLASSES)==label)
    feature = {
      "image": _bytestring_feature([img_bytes]), # one image in the list
      "class": _int_feature([class_num]),        # one class in the list
              }
    return tf.train.Example(features=tf.train.Features(feature=feature))

#-----------------------------------
def resize_and_crop_image(image, label):
    """
    This function
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
def read_image_and_label(img_path):
    """
    This function
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
### TRAINING FUNCTIONS
###############################################################

def weighted_binary_crossentropy(zero_weight, one_weight):
    """
    This function computes weighted binary crossentropy loss
    """
    def wbce(y_true, y_pred):

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred, from_logits=True)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.mean(weighted_b_ce)

    return wbce

# learning rate function
def lrfn(epoch):
    """
    This function creates a custom piecewise linear-exponential learning rate function
    for a custom learning rate scheduler. It is linear to a max, then exponentially decays
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
