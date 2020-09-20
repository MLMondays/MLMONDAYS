

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

###############################################################
## IMPORTS
###############################################################
from imports import *

###############################################################
### DATA FUNCTIONS
###############################################################
#-----------------------------------
def get_training_dataset():
    """
    This function will return a batched dataset for model training
    INPUTS: None
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: training_filenames
    OUTPUTS: batched data set object
    """
    return get_batched_dataset(training_filenames)

def get_validation_dataset():
    """
    This function will return a batched dataset for model training
    INPUTS: None
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: validation_filenames
    OUTPUTS: batched data set object
    """
    return get_batched_dataset(validation_filenames)

def get_validation_eval_dataset():
    """
    This function will return a batched dataset for model training
    INPUTS: None
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: validation_filenames
    OUTPUTS: batched data set object
    """
    return get_eval_dataset(validation_filenames)

#-----------------------------------
def get_batched_dataset(filenames):
    """
    "get_batched_dataset"
    This function defines a workflow for the model to read data from
    tfrecord files by defining the degree of parallelism, batch size, pre-fetching, etc
    and also formats the imagery properly for model training
    (assumes mobilenet by using read_tfrecord_mv2)
    INPUTS:
        * filenames [list]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: BATCH_SIZE, AUTO
    OUTPUTS: tf.data.Dataset object
    """
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = True

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

    dataset = dataset.cache() # This dataset fits in RAM
    #dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(AUTO) #

    return dataset

#-----------------------------------
def get_train_stuff(num_batches):
    """
    "get_train_stuff"
    This function returns all the images and labels from a tf.data.Dataset
    INPUTS:
        * num_batches [int]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * X_train [list] of ndarray images
        * y_train [list] of integer labels
        * class_idx_to_train_idxs [dict] of indices into each class
    """
    X_train = []
    ytrain = []
    train_ds = get_training_dataset()

    counter = 0
    for imgs,lbls in train_ds.take(num_batches):
      ytrain.append(lbls.numpy())
      for im in imgs:
        X_train.append(im.numpy())

    X_train = np.array(X_train)
    ytrain = np.hstack(ytrain)

    # get X_train, y_train arrays
    X_train = X_train.astype("float32")
    ytrain = np.squeeze(ytrain)

    # code repurposed from https://keras.io/examples/vision/metric_learning/
    class_idx_to_train_idxs = defaultdict(list)
    for y_train_idx, y in enumerate(ytrain):
        class_idx_to_train_idxs[y].append(y_train_idx)

    return X_train, ytrain, class_idx_to_train_idxs

#-----------------------------------
def get_test_stuff(num_batches):
    """
    "get_test_stuff"
    This function returns all the images and labels from a tf.data.Dataset
    INPUTS:
        * num_batches [int]
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * X_test [list] of ndarray images
        * y_test [list] of integer labels
        * class_idx_to_test_idxs [dict] of indices into each class
    """
    X_test = []
    ytest = []
    test_ds = get_validation_dataset()

    counter = 0
    for imgs,lbls in test_ds.take(num_batches):
      ytest.append(lbls.numpy())
      for im in imgs:
        X_test.append(im.numpy())

    X_test = np.array(X_test)
    ytest = np.hstack(ytest)

    # get X_test, y_test arrays
    X_test = X_test.astype("float32")
    ytest = np.squeeze(ytest)

    # code repurposed from https://keras.io/examples/vision/metric_learning/
    class_idx_to_test_idxs = defaultdict(list)
    for y_test_idx, y in enumerate(ytest):
        class_idx_to_test_idxs[y].append(y_test_idx)

    return X_test, ytest, class_idx_to_test_idxs

###############################################################
### MODEL FUNCTIONS
###############################################################

class AnchorPositivePairs(tf.keras.utils.Sequence):
    """
    # code modified from https://keras.io/examples/vision/metric_learning/
    "AnchorPositivePairs"
    This Class selects an anchor and positive example images from each label class
    INPUTS: None
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS:
        * x [ndarray]: a pair of example images of each class, (2, num_classes, TARGET_SIZE, TARGET_SIZE, 3)
    """
    def __init__(self, num_batchs):
        self.num_batchs = num_batchs

    def __len__(self):
        return self.num_batchs

    def __getitem__(self, _idx):
        x = np.empty((2, num_classes, TARGET_SIZE, TARGET_SIZE, 3), dtype=np.float32)
        for class_idx in range(num_classes):
            examples_for_class = class_idx_to_train_idxs[class_idx]
            anchor_idx = np.random.choice(examples_for_class)
            positive_idx = np.random.choice(examples_for_class)
            while positive_idx == anchor_idx:
                positive_idx = np.random.choice(examples_for_class)
            x[0, class_idx] = X_train[anchor_idx]
            x[1, class_idx] = X_train[positive_idx]
        return x


###############################################################
## VARIABLES
###############################################################

## 12 classes

## model inputs
json_file = os.getcwd()+os.sep+'data/tamucc/subset_12class/tamucc_subset_12classes.json'

data_path= os.getcwd()+os.sep+"data/tamucc/subset_12class/400"
test_samples_fig = os.getcwd()+os.sep+'results/tamucc_sample_12class_model2_est36samples.png'

cm_filename = os.getcwd()+os.sep+'results/tamucc_sample_12class_model2_cm_val.png'

sample_data_path= os.getcwd()+os.sep+"data/tamucc/subset_12class/sample"

filepath = os.getcwd()+os.sep+'results/tamucc_subset_12class_best_weights_model2.h5'

hist_fig = os.getcwd()+os.sep+'results/tamucc_subset_12class_custom_model2.png'

cm_fig = os.getcwd()+os.sep+'results/tamucc_subset_12class_cm_test_model2.png'

initial_filepath = os.getcwd()+os.sep+'results/tamucc_subset_12class_best_weights_model1.h5'


patience = 10

# double the number of embedding dims

num_embed_dim = 16 #8

# more maximum training epochs, just in case!

max_epochs = 600 #400
lr = 1e-4

n_neighbors = 3

###############################################################
## EXECUTION
###############################################################
filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrec'))

nb_images = ims_per_shard * len(filenames)
print(nb_images)

split = int(len(filenames) * VALIDATION_SPLIT)

training_filenames = filenames[split:]
validation_filenames = filenames[:split]

validation_steps = int(nb_images // len(filenames) * len(validation_filenames)) // BATCH_SIZE
steps_per_epoch = int(nb_images // len(filenames) * len(training_filenames)) // BATCH_SIZE

print(steps_per_epoch)
print(validation_steps)

CLASSES = read_classes_from_json(json_file)

###=============================================

train_ds = get_training_dataset()
val_ds = get_validation_dataset()

#-------------------------------------------------
training_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrec'))
nb_images = ims_per_shard * len(training_filenames)
print(nb_images)

num_batches = int(((1-VALIDATION_SPLIT) * nb_images) / BATCH_SIZE)
print(num_batches)
#
# num_batches = 140 #150

X_train, ytrain, class_idx_to_train_idxs = get_train_stuff(num_batches)

model2 = get_large_embedding_model(TARGET_SIZE, num_classes, num_embed_dim)

# use a smaller learning rate, because we are fine-tuning
lr = 5e-5

patience = 20

model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
     metrics=['accuracy'],
)

# load with previous weights
model2. load_weights(initial_filepath)

earlystop = EarlyStopping(monitor="loss",
                              mode="min", patience=patience)

# set checkpoint file
model_checkpoint = ModelCheckpoint(filepath, monitor='loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)

callbacks = [model_checkpoint, earlystop]


# class_weights = class_weight.compute_class_weight('balanced',
#                                                  np.unique(ytrain),
#                                                  ytrain)

# plt.figure(figsize=(10,10))
# plt.hist(ytrain, bins=np.arange(len(CLASSES)+1), rwidth=.2)
# plt.gca().set_xticks(np.arange(len(CLASSES))+0.5)
# plt.gca().set_xticklabels([c.decode() for c in CLASSES], rotation=90)
# plt.ylabel('Number of images')

#
# class_weights = dict(enumerate(class_weights))
# print(class_weights)

# there is class imbalance, but don't know how to incorporate


do_train = False #True


if do_train:
    history1 = model2.fit(AnchorPositivePairs(num_batchs=num_batches), epochs=max_epochs,
                          callbacks=callbacks)

    plt.figure(figsize = (10,10))
    plt.subplot(221)
    plt.plot(history1.history["loss"])
    plt.xlabel('Model training epoch number')
    plt.ylabel('Loss (soft cosine distance)')

    plt.subplot(222)
    plt.plot(history1.history["accuracy"])
    plt.xlabel('Model training epoch number')
    plt.ylabel('Accuracy')
    # plt.show()
    plt.savefig(hist_fig, dpi=200, bbox_inches='tight')
    plt.close('all')

else:
    model2.load_weights(filepath)


K.clear_session()

#### classify

num_dim_use = num_embed_dim #2

knn3 = fit_knn_to_embeddings(model2, X_train, ytrain, num_dim_use, n_neighbors)

knn5 = fit_knn_to_embeddings(model2, X_train, ytrain, num_dim_use, 5)

knn7 = fit_knn_to_embeddings(model2, X_train, ytrain, num_dim_use, 7)

del X_train, ytrain


## ensemble model

X_test, ytest, class_idx_to_test_idxs = get_test_stuff(num_batches)

touse = len(X_test) #900

# touse = 300

embeddings_test = model2.predict(X_test[:touse])
embeddings_test = tf.nn.l2_normalize(embeddings_test, axis=-1)
del X_test

y_pred1 = knn3.predict(embeddings_test[:,:num_dim_use])
y_pred2 = knn5.predict(embeddings_test[:,:num_dim_use])
y_pred3 = knn7.predict(embeddings_test[:,:num_dim_use])

y_prob1 = knn3.predict_proba(embeddings_test[:,:num_dim_use])
y_prob2 = knn5.predict_proba(embeddings_test[:,:num_dim_use])
y_prob3 = knn7.predict_proba(embeddings_test[:,:num_dim_use])

score1 = knn3.score(embeddings_test[:,:num_dim_use], ytest[:touse])
score2 = knn5.score(embeddings_test[:,:num_dim_use], ytest[:touse])
score3 = knn7.score(embeddings_test[:,:num_dim_use], ytest[:touse])

print('3-NN score: %f' % score1)
print('5-NN score: %f' % score2)
print('7-NN score: %f' % score3)

mask = np.c_[y_pred1, y_pred2, y_pred3]

use = np.any(mask>.9, axis=1) #only predictions where all probabilities are > 0.9
mask = mask[use,:]


# weighted average
# y_en = np.round(np.average(mask, axis=1, weights=[.1, .1, .5]))

y_en = np.median(mask, axis=1)

p_confmat(ytest[:touse][use], y_en, cm_filename.replace('val', 'val_v3_ensemble'), CLASSES, thres = 0.1)


# 0.66
