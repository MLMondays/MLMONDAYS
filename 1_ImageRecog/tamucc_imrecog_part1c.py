
###############################################################
## IMPORTS
###############################################################
from imports import *

###############################################################
## FUNCTIONS
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

## scale up the full data set

## model inputs
data_path= os.getcwd()+os.sep+"data/tamucc/full_2class/400"

test_samples_fig = os.getcwd()+os.sep+'results/tamucc_full_sample_2class_mv2_model2_est24samples.png'

cm_filename = os.getcwd()+os.sep+'results/tamucc_full_sample_2class_mv2_model2_cm_val.png'

sample_data_path= os.getcwd()+os.sep+"data/tamucc/full_2class/sample"

hist_fig = os.getcwd()+os.sep+'results/tamucc_full_sample_2class_custom_model2.png'

filepath = os.getcwd()+os.sep+'results/tamucc_full_2class_custom_best_weights_model2.h5'

CLASSES = [b'dev', b'undev']
patience = 10

# # more data - so we can use a smaller validation split (better fits in memory!)
VALIDATION_SPLIT = 0.4

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

lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

#####################################################################
## class weights
train_ds = get_training_dataset()
val_ds = get_validation_dataset()

numclass = len(CLASSES)
# ID_MAP = dict(zip(np.arange(numclass), [str(k) for k in range(numclass)]))

custom_model3 = make_cat_model(numclass, denseunits=128, base_filters = 30, dropout=0.5) #256


custom_model3.compile(optimizer=tf.keras.optimizers.Adam(),
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
    history = custom_model3.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                          validation_data=val_ds, validation_steps=validation_steps,
                          callbacks=callbacks)

    # Plot training history
    plot_history(history, hist_fig)

    plt.close('all')
    K.clear_session()

else:
    custom_model3.load_weights(filepath)

##########################################################
### evaluate
val_ds = get_validation_eval_dataset()

loss, accuracy = custom_model3.evaluate(val_ds, batch_size=BATCH_SIZE)
print('Test Mean Accuracy: ', round((accuracy)*100, 2),' %')

##86%

##########################################################
### predict

sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))

make_sample_plot(custom_model3, sample_filenames, test_samples_fig, CLASSES)

##################################################

## confusion matrix
val_ds = get_validation_eval_dataset()

labs, preds = get_label_pairs(val_ds, custom_model3)

p_confmat(labs, preds, cm_filename, CLASSES)

#80%
