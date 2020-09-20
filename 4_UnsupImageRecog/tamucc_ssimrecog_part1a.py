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
        X_train.append(im)

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
        X_test.append(im)

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

## model inputs
data_path= os.getcwd().replace('4_UnsupImageRecog', '1_ImageRecog')+os.sep+"data/tamucc/subset_2class/400"
test_samples_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_mv2_model1_est24samples.png'

cm_filename = os.getcwd()+os.sep+'results/tamucc_sample_2class_mv2_model1_cm_val.png'

sample_data_path= os.getcwd().replace('4_UnsupImageRecog', '1_ImageRecog')+os.sep+"data/tamucc/subset_2class/sample"

filepath = os.getcwd()+os.sep+'results/tamucc_subset_2class_custom_best_weights_model1.h5'

hist_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_custom_model1.png'

nn_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_custom_model1_acc_vs_nn.png'

trainsamples_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_trainsamples.png'

valsamples_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_validationsamples.png'

cm_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_cm_test.png'


CLASSES = [b'dev', b'undev']
patience = 10
num_embed_dim = 8
max_epochs = 100
lr = 1e-4

num_classes = len(CLASSES)

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

train_ds = get_training_dataset()
plt.figure(figsize=(16,16))
for imgs,lbls in train_ds.take(1):
  #print(lbls)
  for count,im in enumerate(imgs):
     plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
     plt.imshow(im)
     plt.title(CLASSES[lbls.numpy()[count]], fontsize=8)
     plt.axis('off')
# plt.show()
plt.savefig(trainsamples_fig, dpi=200, bbox_inches='tight')
plt.close('all')


val_ds = get_validation_dataset()
plt.figure(figsize=(16,16))
for imgs,lbls in val_ds.take(1):
  #print(lbls)
  for count,im in enumerate(imgs):
     plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
     plt.imshow(im)
     plt.title(CLASSES[lbls.numpy()[count]], fontsize=8)
     plt.axis('off')
# plt.show()
plt.savefig(valsamples_fig, dpi=200, bbox_inches='tight')
plt.close('all')


training_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrec'))
nb_images = ims_per_shard * len(training_filenames)
print(nb_images)

num_batches = int(((1-VALIDATION_SPLIT) * nb_images) / BATCH_SIZE)
print(num_batches)

num_batches = 10

X_train, ytrain, class_idx_to_train_idxs = get_data_stuff(get_training_dataset(), num_batches)


model = get_embedding_model(TARGET_SIZE, num_classes, num_embed_dim)

num_batches = int(np.ceil(len(X_train) / len(CLASSES)))
print(num_batches)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
     metrics=['accuracy'],
)

earlystop = EarlyStopping(monitor="loss",
                              mode="min", patience=patience)

# set checkpoint file
model_checkpoint = ModelCheckpoint(filepath, monitor='loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)

callbacks = [model_checkpoint, earlystop]

do_train = True # False #True

# no internal validation, so no validation dataset
# what is this dfoing and how does it work

# show embeddings, etc

if do_train:
    history = model.fit(AnchorPositivePairs(num_batchs=num_batches), epochs=max_epochs, callbacks=callbacks)

    plt.figure(figsize = (10,10))
    plt.subplot(221)
    plt.plot(history.history["loss"])
    plt.xlabel('Model training epoch number')
    plt.ylabel('Loss (soft cosine distance)')

    plt.subplot(222)
    plt.plot(history.history["accuracy"])
    plt.xlabel('Model training epoch number')
    plt.ylabel('Accuracy')
    # plt.show()
    plt.savefig(hist_fig, dpi=200, bbox_inches='tight')
    plt.close('all')

else:
    model.load_weights(filepath)


K.clear_session()


#### classify

#we'll use them all, but this number could be as low as 2
num_dim_use = num_embed_dim #i.e. 8

## make functions

knn = fit_knn_to_embeddings(X_train, ytrain, num_dim_use)

del X_train, ytrain

X_test, ytest, class_idx_to_test_idxs = get_data_stuff(get_validation_dataset(), num_batches)

touse = 1000

embeddings_test = model.predict(X_test[:touse])
embeddings_test = tf.nn.l2_normalize(embeddings_test, axis=-1)
del X_test

print('KNN score: %f' % knn.score(embeddings_test[:,:num_dim_use], ytest[:touse]))


## save knn model
## save ann model
# load both and apply


y_pred = knn.predict(embeddings_test[:,:num_dim_use])

# cm = confusion_matrix(ytest[:touse], y_pred, normalize='true'')

p_confmat(ytest[:touse], y_pred, cm_filename, CLASSES, thres = 0.1)


## apply to image files

sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))

for f in sample_filenames[:10]:
    image = file2tensor(f)
    image = tf.cast(image, np.float32)

    embeddings_sample = model.predict(tf.expand_dims(image, 0))

    #knn.predict_proba(embeddings_sample[:,:2])
    obs_class = f.split('/')[-1].split('_IMG')[0]
    est_class = CLASSES[knn.predict(embeddings_sample[:,:num_dim_use])[0]].decode()

    print('pred:%s, est:%s' % (obs_class, est_class ) )


#
# cm = conf_mat_filesamples(model, knn, sample_filenames, num_classes, num_dim_use)
#
# thres = 0.1
# cm[cm<thres] = 0
#
# plt.figure(figsize=(15,15))
# sns.heatmap(cm,
#     annot=True,
#     cmap = sns.cubehelix_palette(dark=0, light=1, as_cmap=True))
#
# tick_marks = np.arange(len(CLASSES))+.5
# plt.xticks(tick_marks, [c.decode() for c in CLASSES], rotation=45,fontsize=12)
# plt.yticks(tick_marks, [c.decode() for c in CLASSES],rotation=45, fontsize=12)
#
# plt.show()
#
#


# advantages:
# 1. fewer train examples? test This
# 2. how does knn predict probabiilty?
# 3. vsiaulize embeddings per sample - just 8 numbers - do this
# 4.







# from sklearn.gaussian_process import GaussianProcessClassifier
# #from sklearn.mixture import GaussianMixture
#
# # X_train, ytrain, class_idx_to_train_idxs = get_train_stuff(num_batches)
#
# embeddings = model.predict(X_train)
# embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
# del X_train
#
# # clf = GaussianMixture().fit(embeddings, ytrain)
#
# clf = GaussianProcessClassifier().fit(embeddings, ytrain)
#
# del embeddings, ytrain
#
#
# X_test, ytest, class_idx_to_test_idxs = get_test_stuff(num_batches)
#
# embeddings_test = model.predict(X_test[:100])
# del X_test
#
# embeddings_test = tf.nn.l2_normalize(embeddings_test, axis=-1)
#
# print('GaussianProcessClassifier score: %f' % clf.score(embeddings_test, ytest[:100]))
#
# sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))
#
# for f in sample_filenames[:100]: #f = sample_filenames[0]
#     image = file2tensor(f)
#     image = tf.cast(image, np.float32)
#
#     embeddings_sample = model.predict(tf.expand_dims(image, 0))
#
#     #clf.predict_proba(embeddings_sample)
#
#     print('pred:%s, est:%s' % (f.split('/')[-1].split('_IMG')[0], CLASSES[clf.predict(embeddings_sample)[0]].decode() ) )
#
#
#
#











#
#
# del X_train
# del ytrain
# # num_batches = 600 = too big for memory
#
# X_test, ytest, class_idx_to_test_idxs = get_test_stuff(num_batches)
#
# gram_matrix = get_gram_matrix(X_test, model)
#
# min_num_near_neighbours = 5
# max_num_near_neighbours = 20
# MN, confusion_matrix = conf_matrix(ytest, gram_matrix, max_num_near_neighbours, min_num_near_neighbours, num_classes, class_idx_to_test_idxs)
#
# plt.plot(np.arange(min_num_near_neighbours,max_num_near_neighbours,1), MN)
# plt.ylabel('Average model accuracy')
# plt.xlabel('Number of nearest neighbours')
# # plt.show()
# plt.savefig(nn_fig, dpi=200, bbox_inches='tight')
# plt.close('all')
#
#
# fig = plt.figure(figsize=(15,15))
# ax = fig.add_subplot(111)
# # Display a confusion matrix.
# disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[c.decode() for c in CLASSES])
# disp.plot(include_values=True, cmap="viridis", ax=ax, xticks_rotation="vertical")
# # plt.show()
# plt.savefig(cm_fig, dpi=200, bbox_inches='tight')
# plt.close('all')
#
#
# near_neighbours_per_example = np.argmax(MN)+1
# print(near_neighbours_per_example)
#
# near_neighbours = near_neighbours_from_samples(X_test, model, near_neighbours_per_example)
#
# #X is much larger than gram_matrix
# #whos ndarray
#
# del X_test
#
# confusion_matrix = get_cm_from_near_neighbours(ytest, num_classes, near_neighbours, near_neighbours_per_example, class_idx_to_test_idxs)
#
#
# fig = plt.figure(figsize=(15,15))
# ax = fig.add_subplot(111)
# # Display a confusion matrix.
# disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[c.decode() for c in CLASSES])
# disp.plot(include_values=True, cmap="viridis", ax=ax, xticks_rotation="vertical")
# # plt.show()
# plt.savefig(cm_fig.replace('test', 'test_'+str(near_neighbours_per_example)+'_neighbours'), dpi=200, bbox_inches='tight')
# plt.close('all')
#
#
#
#
# min_neighbours = np.min(np.argsort(MN)[:-5:-1])
# max_neighbours = np.max(np.argsort(MN)[:-5:-1])
#
# CM = []
#
# for near_neighbours_per_example in np.arange(min_neighbours,max_neighbours,1):
#
#   near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]
#   confusion_matrix = np.zeros((num_classes, num_classes))
#
#   # For each class.
#   for class_idx in range(num_classes):
#       # Consider 10 examples.
#       example_idxs = class_idx_to_test_idxs[class_idx][:near_neighbours_per_example] #[:10]
#       for y_test_idx in example_idxs:
#           # And count the classes of its near neighbours.
#           for nn_idx in near_neighbours[y_test_idx][:-1]:
#               #print("dist: %f, class: %i" % (gram_matrix[nn_idx, nn_idx], y_test[nn_idx]))
#               nn_class_idx = ytest[nn_idx]
#               confusion_matrix[class_idx, nn_class_idx] += 1
#   CM.append(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])
#
#
# cm = np.mean(CM, axis=0)
#
# fig = plt.figure(figsize=(15,15))
# ax = fig.add_subplot(111)
# # Display a confusion matrix.
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
# disp.plot(include_values=True, cmap="viridis", ax=ax, xticks_rotation="vertical")
# plt.title("Ensemble confusion matrix")
#
# print("Mean accuracy: %f" % (np.mean(np.diag(cm))))
#
# # plt.show()
# plt.savefig(cm_fig.replace('test', 'test_ensemble'), dpi=200, bbox_inches='tight')
# plt.close('all')
#
#
# cmv2 = np.std(CM, axis=0) / np.mean(CM, axis=0)
#
# plt.figure(figsize=(10,10))
# ax = plt.subplot(111)
# ax.plot(100*np.diag(cmv2), 'k-o')
# ax.set_xticks(np.arange(len(CLASSES)))
# ax.set_xticklabels(CLASSES, fontsize=10, rotation=30)
# plt.ylabel('Variability Index, %')
#
# # plt.show()
# plt.savefig(cm_fig.replace('test', 'test_ensemble_variability'), dpi=200, bbox_inches='tight')
# plt.close('all')
