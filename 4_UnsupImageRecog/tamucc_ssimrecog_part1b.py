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

## model inputs
data_path= os.getcwd().replace('4_UnsupImageRecog', '1_ImageRecog')+os.sep+"data/tamucc/subset_2class/400"
test_samples_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_mv2_model2_est24samples.png'

cm_filename = os.getcwd()+os.sep+'results/tamucc_sample_2class_mv2_model2_cm_val.png'

sample_data_path= os.getcwd().replace('4_UnsupImageRecog', '1_ImageRecog')+os.sep+"data/tamucc/subset_2class/sample"

filepath = os.getcwd()+os.sep+'results/tamucc_subset_2class_custom_best_weights_model2.h5'

hist_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_custom_model2.png'

nn_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_custom_model2_acc_vs_nn.png'

trainsamples_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_trainsamples.png'

valsamples_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_validationsamples.png'

cm_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_cm_test.png'

CLASSES = [b'dev', b'undev']
patience = 10

num_embed_dim =8

max_epochs = 100
lr = 1e-4

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

train_ds = get_training_dataset()

val_ds = get_validation_dataset()

training_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrec'))
nb_images = ims_per_shard * len(training_filenames)
print(nb_images)

num_batches = int(((1-VALIDATION_SPLIT) * nb_images) / BATCH_SIZE)
print(num_batches)

num_batches = 200

X_train, ytrain, class_idx_to_train_idxs  = get_data_stuff(train_ds, num_batches)


#####################################################################
## class weights
print('.....................................')
print('Computing class weights ...')

l = []
num_batches = int(((1-VALIDATION_SPLIT) * nb_images) / BATCH_SIZE)
# train_ds = get_training_dataset()
for _,lbls in train_ds.take(num_batches):
    l.append(lbls.numpy())

# val_ds = get_validation_dataset()
num_batches = int(((VALIDATION_SPLIT) * nb_images) / BATCH_SIZE)
for _,lbls in val_ds.take(num_batches):
    l.append(lbls.numpy())

l = np.asarray(l).flatten()

# class weights will be given by n_samples / (n_classes * np.bincount(y))

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(l),
                                                 l)

class_weights = np.round(class_weights)
class_weights /= np.sum(class_weights)

# class_weights = dict(enumerate(class_weights))
print(class_weights)


# weight the loss function directly (i.e. without passing 'class weights' to the .fit() command)
# by making a custom loss function that computes binary crossentrpy then weights by a measure of the inverse relative proportion of the class
print('.....................................')
print('Creating and compiling model ...')


model2 = get_embedding_model(TARGET_SIZE, num_classes, num_embed_dim)

num_batches = int(np.ceil(len(X_train) / len(CLASSES)))
print(num_batches)



model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
#     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    # loss=weighted_binary_crossentropy(class_weights[0], class_weights[1]),
    loss=weighted_binary_crossentropy(0.7, 0.3),
     metrics=['accuracy'],
)

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
    history2 = model2.fit(AnchorPositivePairs(num_batchs=num_batches), epochs=max_epochs,
                          callbacks=callbacks)

    plt.figure(figsize = (10,10))
    plt.subplot(221)
    plt.plot(history2.history["loss"])
    plt.xlabel('Model training epoch number')
    plt.ylabel('Loss (soft cosine distance)')

    plt.subplot(222)
    plt.plot(history2.history["accuracy"])
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

## make functions
print('.....................................')
print('Fitting kNN model to embeddings ...')
n_neighbours = 3

knn2 = fit_knn_to_embeddings(model2, X_train, ytrain, n_neighbours)

del X_train, ytrain

num_batches = 100

print('.....................................')
print('Evaluating model ...')
X_test, ytest, class_idx_to_test_idxs = get_data_stuff(val_ds, num_batches)

touse = 1000

embeddings_test = model2.predict(X_test[:touse])
embeddings_test = tf.nn.l2_normalize(embeddings_test, axis=-1)
del X_test

print('KNN score: %f' % knn2.score(embeddings_test[:,:num_dim_use], ytest[:touse]))


y_pred = knn2.predict(embeddings_test[:,:num_dim_use])

p_confmat(ytest[:touse], y_pred, cm_filename, CLASSES, thres = 0.1)

print('.....................................')
print('Using model for prediction on jpeg images ...')
## apply to image files

sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))

for f in sample_filenames[:10]:
    image = file2tensor(f)
    image = tf.cast(image, np.float32)

    embeddings_sample = model2.predict(tf.expand_dims(image, 0))

    obs_class = f.split('/')[-1].split('_IMG')[0]
    est_class = CLASSES[knn2.predict(embeddings_sample[:,:num_dim_use])[0]].decode()

    print('pred:%s, est:%s' % (obs_class, est_class ) )
