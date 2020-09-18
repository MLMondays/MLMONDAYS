
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

#-----------------------------------
def get_all_labels(nb_images, VALIDATION_SPLIT, BATCH_SIZE):
    """
    "get_all_labels"
    This function will obtain the classes of all samples in both train and
    validation sets. For computing class imbalance on the whole dataset
    INPUTS:
        * nb_images [int]: number of total images
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: VALIDATION_SPLIT, BATCH_SIZE
    OUTPUTS:
        * l [list]: list of integers representing labels of each image
    """
    l = []
    num_batches = int(((1-VALIDATION_SPLIT) * nb_images) / BATCH_SIZE)
    train_ds = get_training_dataset()
    for _,lbls in train_ds.take(num_batches):
        l.append(lbls.numpy())

    val_ds = get_validation_dataset()
    num_batches = int(((VALIDATION_SPLIT) * nb_images) / BATCH_SIZE)
    for _,lbls in val_ds.take(num_batches):
        l.append(lbls.numpy())

    l = np.asarray(l).flatten()
    return l

###############################################################
## VARIABLES
###############################################################


data_path= os.getcwd()+os.sep+"data/tamucc/full_4class/400"

json_file = os.getcwd()+os.sep+'data/tamucc/full_4class/tamucc_full_4classes.json'

filepath = os.getcwd()+os.sep+'results/tamucc_full_4class_mv2_best_weights_model3.h5'

train_hist_fig = os.getcwd()+os.sep+'results/tamucc_full_4class_mv2_model3.png'

cm_filename = os.getcwd()+os.sep+'results/tamucc_full_4class_mv2_model3_cm_val.png'
sample_plot_name = os.getcwd()+os.sep+'results/tamucc_full_4class_mv2_model3_est24samples.png'

sample_data_path= os.getcwd()+os.sep+"data/tamucc/full_4class/sample"

## transfer learning
initial_weights = os.getcwd()+os.sep+'results/tamucc_subset_4class_mv2_best_weights_model2.h5'

test_samples_fig = os.getcwd()+os.sep+'results/tamucc_full_sample_4class_mv2_model3_est24samples.png'

patience = 30

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
# augmented_train_ds, augmented_val_ds = get_aug_datasets()

plt.figure(figsize=(12,12))
for imgs,lbls in train_ds.take(1):
  #print(lbls)
  for count,im in enumerate(imgs):
     plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
     plt.imshow(im)
     plt.title(CLASSES[lbls.numpy()[count]], fontsize=8)
     plt.axis('off')
# plt.show()
plt.savefig(os.getcwd()+os.sep+'results/tamucc_full_4class_trainsamples.png', dpi=200, bbox_inches='tight')


plt.figure(figsize=(12,12))
for imgs,lbls in val_ds.take(1):
  #print(lbls)
  for count,im in enumerate(imgs):
     plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
     plt.imshow(im)
     plt.title(CLASSES[lbls.numpy()[count]], fontsize=8)
     plt.axis('off')
# plt.show()
plt.savefig(os.getcwd()+os.sep+'results/tamucc_full_4class_valsamples.png', dpi=200, bbox_inches='tight')

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

model = transfer_learning_mobilenet_model(len(CLASSES), (TARGET_SIZE, TARGET_SIZE, 3), dropout_rate=0.5)

model.compile(optimizer=tf.keras.optimizers.Adam(), #1e-4),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])


## transfer learning
model.load_weights(initial_weights)

earlystop = EarlyStopping(monitor="val_loss",
                              mode="min", patience=patience)

# set checkpoint file
model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)

callbacks = [model_checkpoint, earlystop, lr_callback]

do_train = True #False #True

# model.summary()

if do_train:

    ## class weights
    # history = model.fit(augmented_train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
    #                       validation_data=augmented_val_ds, validation_steps=validation_steps,
    #                       callbacks=callbacks, class_weight = class_weights)

    history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                          validation_data=val_ds, validation_steps=validation_steps,
                          callbacks=callbacks, class_weight = class_weights)

    # Plot training history
    plot_history(history, train_hist_fig)

    plt.close('all')
    K.clear_session()

else:
    model.load_weights(filepath)


##########################################################
### evaluate

val_ds = get_validation_eval_dataset()

loss, accuracy = model.evaluate(val_ds, batch_size=BATCH_SIZE)
print('Test Mean Accuracy: ', round((accuracy)*100, 2),' %')

##92%

##########################################################
### predict

sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))

make_sample_plot(model, sample_filenames, test_samples_fig, CLASSES)


##################################################

## confusion matrix
val_ds = get_validation_eval_dataset()

labs, preds = get_label_pairs(val_ds, model)

p_confmat(labs, preds, cm_filename, CLASSES)

#86%
