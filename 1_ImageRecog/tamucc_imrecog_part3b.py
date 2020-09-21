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
def get_aug_datasets():
    """
    This function will create train and validation sets based on a specific
    data augmentation pipeline consisting of random flipping, small rotations,
    translations and contrast adjustments
    INPUTS: None
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: validation_filenames, training_filenames
    OUTPUTS: two batched data set objects, one for training and one for validation
    """
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.01),
      tf.keras.layers.experimental.preprocessing.RandomTranslation(0.1,0.1),
      tf.keras.layers.experimental.preprocessing.RandomContrast(0.1)
    ])

    augmented_train_ds = get_training_dataset().map(
      lambda x, y: (data_augmentation(x, training=True), y))

    augmented_val_ds = get_validation_dataset().map(
      lambda x, y: (data_augmentation(x, training=True), y))
    return augmented_train_ds, augmented_val_ds


###############################################################
## VARIABLES
###############################################################

data_path= os.getcwd()+os.sep+"data/tamucc/subset_4class/400"

json_file = os.getcwd()+os.sep+'data/tamucc/subset_4class/tamucc_subset_4classes.json'

filepath = os.getcwd()+os.sep+'results/tamucc_subset_4class_mv2_best_weights_model2.h5'

train_hist_fig = os.getcwd()+os.sep+'results/tamucc_sample_4class_mv2_model2.png'

cm_filename = os.getcwd()+os.sep+'results/tamucc_sample_4class_mv2_model2_cm_val.png'
sample_plot_name = os.getcwd()+os.sep+'results/tamucc_sample_4class_mv2_model2_est24samples.png'

sample_data_path= os.getcwd()+os.sep+"data/tamucc/subset_4class/sample"

test_samples_fig = os.getcwd()+os.sep+'results/tamucc_full_sample_4class_mv2_model2_est24samples.png'

###############################################################
## EXECUTION
###############################################################

filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrec'))

CLASSES = read_classes_from_json(json_file)
print(CLASSES)

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

## data augmentation is typically used

augmented_train_ds, augmented_val_ds = get_aug_datasets()

#####################################################################
## class weights
l = get_all_labels(nb_images, VALIDATION_SPLIT, BATCH_SIZE)

# class weights will be given by n_samples / (n_classes * np.bincount(y))

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(l),
                                                 l)

class_weights = dict(enumerate(class_weights))
print(class_weights)

##=========
lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

model2 = transfer_learning_mobilenet_model(len(CLASSES), (TARGET_SIZE, TARGET_SIZE, 3), dropout_rate=0.5)

model2.compile(optimizer=tf.keras.optimizers.Adam(), #1e-4),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

patience = 30
earlystop = EarlyStopping(monitor="val_loss",
                              mode="min", patience=patience)

# set checkpoint file
model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)

callbacks = [model_checkpoint, earlystop, lr_callback]

do_train = False #True

# model2.summary()

if do_train:

    ## class weights
    history = model2.fit(augmented_train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                          validation_data=augmented_val_ds, validation_steps=validation_steps,
                          callbacks=callbacks, class_weight = class_weights)

    # Plot training history
    plot_history(history, train_hist_fig)

    plt.close('all')
    K.clear_session()

else:
    model2.load_weights(filepath)

##########################################################
### evaluate
loss, accuracy = model2.evaluate(get_validation_eval_dataset(), batch_size=BATCH_SIZE)
print('Test Mean Accuracy: ', round((accuracy)*100, 2),' %')

##89%

##########################################################
### predict
sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))

make_sample_plot(model2, sample_filenames, test_samples_fig, CLASSES)


##################################################
## confusion matrix
val_ds = get_validation_eval_dataset()

labs, preds = get_label_pairs(val_ds, model2)

p_confmat(labs, preds, cm_filename, CLASSES)

#77%
