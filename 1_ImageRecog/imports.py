

## TAMUCC
#from tamucc_imports import *

##UNCOMMENT BELOW TO USE NWPU DATA
from nwpu_imports import *

#see mlmondays blog post:
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

##calcs
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import Xception
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

## plots
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

##i/o
from matplotlib.image import imread
import pandas as pd
import json, shutil

##utils
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

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
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])

    class_label = tf.cast(example['class'], tf.int32)

    return image, class_label

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
### PLOTTING FUNCTIONS
###############################################################

#-----------------------------------
def plot_history(history, train_hist_fig):
    """
    This function
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
### DATASET FUNCTIONS
###############################################################

#-----------------------------------
def file2tensor(f, model='mobilenet'):
    """
    This function
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
    This function
    """
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False  ##True?

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
    This function
    """
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

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
    This function
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
    This function
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
def conv_block(inp, filters=32, bn=True, pool=True): #, drop=True):
   """
   This function generates a convolutional block
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
def make_cat_model(ID_MAP, dropout, denseunits, base_filters, bn=False, pool=True, drop=False, shallow=True):
    """
    This function creates an implementation of a convolutional deep learning model for estimating
	a discrete category
    """
    #base_filters = 30
    input_layer = tf.keras.layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))

    x = conv_block(input_layer, filters=base_filters, bn=bn, pool=pool)#, drop=drop) #x #
    x = conv_block(x, filters=base_filters*2, bn=bn, pool=pool)#, drop=drop)
    x = conv_block(x, filters=base_filters*3, bn=bn, pool=pool)#, drop=drop)
    x = conv_block(x, filters=base_filters*4, bn=bn, pool=pool)#, drop=drop)

    if shallow is False:
        x = conv_block(x, filters=base_filters*5, bn=bn, pool=pool)
        x = conv_block(x, filters=base_filters*6, bn=bn, pool=pool)

    bottleneck = tf.keras.layers.GlobalMaxPool2D()(x)
    bottleneck = tf.keras.layers.Dropout(dropout)(bottleneck)

    # for class prediction
    class_head = tf.keras.layers.Dense(units=denseunits, activation='relu')(bottleneck)  ##128
    class_head = tf.keras.layers.Dense(units=len(ID_MAP), activation='softmax', name='output')(class_head)

    model = tf.keras.models.Model(inputs=input_layer, outputs=[class_head])

    model.compile(optimizer='adam', #'adam',
              loss={'output': 'categorical_crossentropy'}, #
              metrics={'output': 'accuracy'})

    #model.summary()
    return model
