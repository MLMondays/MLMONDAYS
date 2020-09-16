
###############################################################
## IMPORTS
###############################################################
from imports import *

#-----------------------------------
def get_training_dataset():
  return get_batched_dataset(training_filenames)

#-----------------------------------
def get_validation_dataset():
  return get_batched_dataset(validation_filenames)

def get_validation_eval_dataset():
  return get_eval_dataset(validation_filenames)

#-----------------------------------
def get_aug_datasets():

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

## model inputs
data_path= os.getcwd()+os.sep+"data/tamucc/subset_2class/400"

test_samples_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_mv2_model2_est24samples.png'

cm_filename = os.getcwd()+os.sep+'results/tamucc_sample_2class_mv2_model2_cm_val.png'

sample_data_path= os.getcwd()+os.sep+"data/tamucc/subset_2class/sample"

hist_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_custom_model2.png'

filepath = os.getcwd()+os.sep+'results/tamucc_subset_2class_custom_best_weights_model2.h5'

CLASSES = [b'dev', b'undev']
patience = 10


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


## data augmentation is typically used
augmented_train_ds, augmented_val_ds = get_aug_datasets()


lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

##smaller min learning rate (starting from scratch)

#####################################################################
## class weights

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

# class weights will be given by n_samples / (n_classes * np.bincount(y))

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(l),
                                                 l)

class_weights = dict(enumerate(class_weights))
print(class_weights)

numclass = len(CLASSES)
ID_MAP = dict(zip(np.arange(numclass), [str(k) for k in range(numclass)]))


custom_model2 = make_cat_model(ID_MAP, denseunits=256, base_filters = 30, dropout=0.5)


custom_model2.compile(optimizer=tf.keras.optimizers.Adam(),
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
    history = custom_model2.fit(augmented_train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                          validation_data=augmented_val_ds, validation_steps=validation_steps,
                          callbacks=callbacks, class_weight = class_weights)

    # Plot training history
    plot_history(history, hist_fig)
    # plt.show()
    # plt.savefig(hist_fig, dpi=200, bbox_inches='tight')

    plt.close('all')
    K.clear_session()

else:
    custom_model2.load_weights(filepath)


##########################################################
### evaluate
loss, accuracy = custom_model2.evaluate(get_validation_eval_dataset(), batch_size=BATCH_SIZE)
print('Test Mean Accuracy: ', round((accuracy)*100, 2),' %')

##86%


##########################################################
### predict

sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))


plt.figure(figsize=(16,16))

for counter,f in enumerate(sample_filenames):
    image, im = file2tensor(f, 'mobilenet')
    plt.subplot(6,4,counter+1)
    name = sample_filenames[counter].split(os.sep)[-1].split('_')[0]
    plt.title(name, fontsize=10)
    plt.imshow(tf.cast(image, tf.uint8))
    plt.axis('off')

    scores = custom_model2.predict(tf.expand_dims(im, 0) , batch_size=1)
    n = np.argmax(scores[0])
    est_name = CLASSES[n].decode()
    if name==est_name:
       plt.text(10,50,'prediction: %s' % est_name,
                color='k', fontsize=12,
                ha="center", va="center",
                bbox=dict(boxstyle="round",
                       ec=(.1, 1., .5),
                       fc=(.1, 1., .5),
                       ))
    else:
       plt.text(10,50,'prediction: %s' % est_name,
                color='k', fontsize=12,
                ha="center", va="center",
                bbox=dict(boxstyle="round",
                       ec=(1., 0.5, 0.1),
                       fc=(1., 0.8, 0.8),
                       ))

# plt.show()
plt.savefig(test_samples_fig,
            dpi=200, bbox_inches='tight')
plt.close('all')


##################################################
## confusion matrix

## confusion matrix
val_ds = get_validation_eval_dataset()

labs, preds = get_label_pairs(val_ds, custom_model2)

p_confmat(labs, preds, cm_filename, CLASSES)

#80%
