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


###############################################################
## VARIABLES
###############################################################

data_path= os.getcwd()+os.sep+"data/tamucc/subset_3class/400"

sample_data_path= os.getcwd()+os.sep+"data/tamucc/subset_3class/sample"

CLASSES = [b'marsh', b'dev', b'other']

patience = 10

filepath = os.getcwd()+os.sep+'results/tamucc_subset_3class_mv2_best_weights_model2.h5'

train_hist_fig = os.getcwd()+os.sep+'results/tamucc_sample_3class_mv2_model2.png'
cm_filename = os.getcwd()+os.sep+'results/tamucc_sample_3class_mv2_model2_cm_val.png'
sample_plot_name = os.getcwd()+os.sep+'results/tamucc_sample_3class_mv2_model2_est24samples.png'

test_samples_fig = os.getcwd()+os.sep+'results/tamucc_full_sample_3class_mv2_model_est24samples.png'


###############################################################
## EXECUTION
###############################################################

#images already shuffled

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

## data augmentation is typically used
augmented_train_ds, augmented_val_ds = get_aug_datasets()


###==============================================================================
## class imbalance
## two important questions:
## are train and validation sets approximately equal in terms of their class representation?
## are classes imbalanced overall?

## are train and validation sets approximately equal in terms of their class representation?

num_batches = int(((1-VALIDATION_SPLIT) * nb_images) / BATCH_SIZE)
print(num_batches)
Ntrain = []
train_ds = get_training_dataset()
for _,lbls in train_ds.take(num_batches):
  n = np.bincount(lbls, minlength=len(CLASSES))
  Ntrain.append(n)

num_batches = int((VALIDATION_SPLIT * nb_images) / BATCH_SIZE)
print(num_batches)
Nval = []
val_ds = get_validation_dataset()
for _,lbls in val_ds.take(num_batches):
  n = np.bincount(lbls, minlength=len(CLASSES))
  Nval.append(n)

Ntrain = np.sum(Ntrain, axis=0)
Nval = np.sum(Nval, axis=0)

plt.figure(figsize=(10,10))
plt.subplot(121)
plt.loglog(Ntrain, Nval, 'ko')
xl = plt.xlim()
plt.plot(xl,xl,'r--', lw=2)
plt.xlabel('Number of images per class, training set')
plt.ylabel('Number of images per class, validation set')

plt.subplot(122)
plt.plot(Ntrain/np.sum(Ntrain), Nval/np.sum(Nval), 'ko')
xl = plt.xlim()
plt.plot(xl,xl,'r--', lw=2)
plt.xlabel('Proportion of images per class, training set')
plt.ylabel('Proportion of images per class, validation set')

outlier = np.argmax(np.abs(Ntrain/np.sum(Ntrain) - Nval/np.sum(Nval)))
plt.text(Ntrain[outlier]/np.sum(Ntrain), Nval[outlier]/np.sum(Nval), CLASSES[outlier])

# plt.show()
plt.savefig(os.getcwd()+os.sep+'results/tamucc_sample_3class_prop_per_class.png', dpi=200, bbox_inches='tight')


plt.close('all')


N = Ntrain + Nval

plt.figure(figsize=(12,4))
plt.subplot(211)
plt.bar(CLASSES, N)
plt.xticks(rotation=90, fontsize=7)
# plt.show()
plt.savefig(os.getcwd()+os.sep+'results/tamucc_sample_3class_imbalance.png', dpi=200, bbox_inches='tight')

plt.close('all')


########### transfer learning

lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

rng = [i for i in range(MAX_EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, [lrfn(x) for x in rng])
# plt.show()
plt.savefig(os.getcwd()+os.sep+'results/learnratesched.png', dpi=200, bbox_inches='tight')

model2 = transfer_learning_mobilenet_model(len(CLASSES), (TARGET_SIZE, TARGET_SIZE, 3), dropout_rate=0.5)

model2.compile(optimizer=tf.keras.optimizers.Adam(), #1e-4),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

earlystop = EarlyStopping(monitor="val_loss",
                              mode="min", patience=patience)

# set checkpoint file
model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)

callbacks = [model_checkpoint, earlystop, lr_callback]

do_train = False #True

if do_train:

    history = model2.fit(augmented_train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                          validation_data=augmented_val_ds, validation_steps=validation_steps,
                          callbacks=callbacks)

    # Plot training history
    plot_history(history, train_hist_fig)

    # plt.show()
    #plt.savefig(train_hist_fig, dpi=200, bbox_inches='tight')

    plt.close('all')
    K.clear_session()

else:
    model2.load_weights(filepath)


##########################################################
### evaluate
loss, accuracy = model2.evaluate(get_validation_eval_dataset(), batch_size=BATCH_SIZE)
print('Test Mean Accuracy: ', round((accuracy)*100, 2),' %')

##77%

##########################################################
### predict
sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))

make_sample_plot(model2, sample_filenames, test_samples_fig, CLASSES)


## confusion matrix
val_ds = get_validation_eval_dataset()

labs, preds = get_label_pairs(val_ds, model2)

p_confmat(labs, preds, cm_filename, CLASSES)

#73%
