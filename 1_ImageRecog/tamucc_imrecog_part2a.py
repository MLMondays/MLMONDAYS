

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


###############################################################
## VARIABLES
###############################################################

data_path= os.getcwd()+os.sep+"data/tamucc/subset_3class/400"

CLASSES = [b'marsh', b'dev', b'other']

filepath = os.getcwd()+os.sep+'results/tamucc_subset_3class_custom_best_weights_model1.h5'

train_hist_fig = os.getcwd()+os.sep+'results/tamucc_sample_3class_custom_model1.png'
sample_data_path= os.getcwd()+os.sep+"data/tamucc/subset_2class/sample"
cm_filename = os.getcwd()+os.sep+'results/tamucc_sample_3class_custom_model_cm_val.png'
sample_plot_name = os.getcwd()+os.sep+'results/tamucc_sample_3class_custom_model_est24samples.png'


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

print(steps_per_epoch)
print(validation_steps)

train_ds = get_training_dataset()
for imgs,lbls in train_ds.take(1):
  #print(lbls)
  for count,im in enumerate(imgs):
     plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
     plt.imshow(im)
     plt.title(CLASSES[lbls.numpy()[count]], fontsize=8)
     plt.axis('off')
# plt.show()
plt.savefig(os.getcwd()+os.sep+'results/tamucc_sample_3class_trainsamples.png', dpi=200, bbox_inches='tight')


val_ds = get_validation_dataset()
for imgs,lbls in val_ds.take(1):
  #print(lbls)
  for count,im in enumerate(imgs):
     plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
     plt.imshow(im)
     plt.title(CLASSES[lbls.numpy()[count]], fontsize=8)
     plt.axis('off')
# plt.show()
plt.savefig(os.getcwd()+os.sep+'results/tamucc_sample_3class_valsamples.png', dpi=200, bbox_inches='tight')



## data augmentation is typically used
## the primary purpose is regularization
#
# from: https://www.tensorflow.org/tutorials/images/data_augmentation
# With this approach, you use Dataset.map to create a dataset that yields batches of augmented images. In this case:
#
#     Data augmentation will happen asynchronously on the CPU, and is non-blocking. You can overlap the training of your model on the GPU with data preprocessing, using Dataset.prefetch, shown below.
#     In this case the prepreprocessing layers will not be exported with the model when you call model.save. You will need to attach them to your model before saving it or reimplement them server-side. After training, you can attach the preprocessing layers before export.

## data augmentation is typically used
augmented_train_ds, augmented_val_ds = get_aug_datasets()


plt.figure(figsize=(16,16))
for im,l in augmented_train_ds.take(1):
    for count,im in enumerate(im):
       plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
       plt.imshow(im)
       plt.title(CLASSES[l[count]], fontsize=8)
       plt.axis('off')
# plt.show()
plt.savefig(os.getcwd()+os.sep+'results/tamucc_sample_3class_augsamples.png', dpi=200, bbox_inches='tight')



lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

##smaller min learning rate (starting from scratch)

###+===================================================

## smaller model
numclass = len(CLASSES)
ID_MAP = dict(zip(np.arange(numclass), [str(k) for k in range(numclass)]))


# more classes = larger model
# shallow=False

custom_model = make_cat_model(ID_MAP, denseunits=256, base_filters = 30, dropout=0.5, bn=False, pool=True, shallow=False)

custom_model.summary()

custom_model.compile(optimizer=tf.keras.optimizers.Adam(),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])

earlystop = EarlyStopping(monitor="val_loss",
                              mode="min", patience=patience)

# set checkpoint file
model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)

callbacks = [model_checkpoint, earlystop, lr_callback]

# model_json = custom_model.to_json()
# with open(filepath.replace('.h5','.json'), "w") as json_file:
#    json_file.write(model_json)

#618k parameters (small for deep learning)

do_train = False #True

if do_train:
    history = custom_model.fit(augmented_train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                          validation_data=augmented_val_ds, validation_steps=validation_steps,
                          callbacks=callbacks)

    # Plot training history
    plot_history(history, train_hist_fig)

    # plt.show()
    plt.savefig(train_hist_fig, dpi=200, bbox_inches='tight')

    plt.close('all')
    K.clear_session()

else:
    custom_model.load_weights(filepath)


##########################################################
### evaluate
loss, accuracy = custom_model.evaluate(get_validation_eval_dataset(), batch_size=BATCH_SIZE)
print('Test Mean Accuracy: ', round((accuracy)*100, 2),' %')

##76%


##########################################################
### predict
sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))


plt.figure(figsize=(16,16))

for counter,f in enumerate(sample_filenames):
    image, im = file2tensor(f, 'mobilenet')
    plt.subplot(8,4,counter+1)
    name = sample_filenames[counter].split(os.sep)[-1].split('_')[0]
    plt.title(name, fontsize=10)
    plt.imshow(tf.cast(image, tf.uint8))
    plt.axis('off')

    scores = custom_model.predict(tf.expand_dims(im, 0) , batch_size=1)
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
plt.savefig(sample_plot_name,
            dpi=200, bbox_inches='tight')
plt.close('all')




##################################################

## confusion matrix
val_ds = get_validation_eval_dataset()

labs, preds = get_label_pairs(val_ds, custom_model)

p_confmat(labs, preds, cm_filename, CLASSES)

#72%
