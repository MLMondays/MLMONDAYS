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
test_samples_fig = os.getcwd()+os.sep+'results/tamucc_sample_12class_model1_est36samples.png'

cm_filename = os.getcwd()+os.sep+'results/tamucc_sample_12class_model1_cm_val.png'

sample_data_path= os.getcwd()+os.sep+"data/tamucc/subset_12class/sample"

filepath = os.getcwd()+os.sep+'results/tamucc_subset_12class_best_weights_model1.h5'

hist_fig = os.getcwd()+os.sep+'results/tamucc_subset_12class_custom_model1.png'

trainsamples_fig = os.getcwd()+os.sep+'results/tamucc_subset_12class_trainsamples.png'

valsamples_fig = os.getcwd()+os.sep+'results/tamucc_subset_12class_validationsamples.png'

cm_fig = os.getcwd()+os.sep+'results/tamucc_subset_12class_cm_test.png'

patience = 10

# double the number of embedding dims

num_embed_dim = 16 #8

# more maximum training epochs, just in case!

max_epochs = 600 #400
lr = 1e-4

n_neighbours = 3

###############################################################
## EXECUTION
###############################################################
filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrec'))

print('.....................................')
print('Reading files and making datasets ...')

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


train_ds = get_training_dataset()

print('.....................................')
print('Printing examples to file ...')

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

print('.....................................')
print('Reading files and making datasets ...')

#-------------------------------------------------
training_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrec'))
nb_images = ims_per_shard * len(training_filenames)
print(nb_images)

num_batches = int(((1-VALIDATION_SPLIT) * nb_images) / BATCH_SIZE)
print(num_batches)


num_batches = 200

X_train, ytrain, class_idx_to_train_idxs  = get_data_stuff(train_ds, num_batches)

print('.....................................')
print('Creating and compiling model ...')
model1 = get_large_embedding_model(TARGET_SIZE, num_classes, num_embed_dim)

## use SparseCategoricalCrossentropy because multiclass

model1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
     metrics=['accuracy'],
)

model1.summary()

#393k

earlystop = EarlyStopping(monitor="loss",
                              mode="min", patience=patience)

# set checkpoint file
model_checkpoint = ModelCheckpoint(filepath, monitor='loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)

callbacks = [model_checkpoint, earlystop]

do_train = False #True


if do_train:
    print('.....................................')
    print('Training model ...')
    history1 = model1.fit(AnchorPositivePairs(num_batchs=num_batches), epochs=max_epochs,
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
    model1.load_weights(filepath)


K.clear_session()

#### classify
print('.....................................')
print('Creating kNN model ...')

num_dim_use = num_embed_dim #2

knn3 = fit_knn_to_embeddings(model1, X_train, ytrain, n_neighbours)

del X_train, ytrain


print('.....................................')
print('Evaluating model ...')
num_batches = 100

X_test, ytest, class_idx_to_test_idxs = get_data_stuff(val_ds, num_batches)


touse = len(X_test) #900

# touse = 300

embeddings_test = model1.predict(X_test[:touse])
embeddings_test = tf.nn.l2_normalize(embeddings_test, axis=-1)
del X_test

print('KNN score: %f' % knn3.score(embeddings_test[:,:num_dim_use], ytest[:touse]))

## 0.78

y_pred = knn3.predict(embeddings_test[:,:num_dim_use])

p_confmat(ytest[:touse], y_pred, cm_filename, CLASSES, thres = 0.1)

# 0.73


y_pred = knn3.predict_proba(embeddings_test[:,:num_dim_use])

y_prob = np.max(y_pred, axis=1)

y_pred = np.argmax(y_pred, axis=1)

ind = np.where(y_prob==1)[0]

print(len(ind))

p_confmat(ytest[:touse][ind], y_pred[ind], cm_filename.replace('val', 'val_v2'), CLASSES, thres = 0.1)

# 0.81
