

###############################################################
## IMPORTS
###############################################################
from imports import *

print(TARGET_SIZE)


#-----------------------------------
def get_training_dataset():
  return get_batched_dataset(training_filenames)

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

#-----------------------------------
def read_image_and_label(img_path):

  bits = tf.io.read_file(img_path)
  image = tf.image.decode_jpeg(bits)

  label = tf.strings.split(img_path, sep='/')
  # remove
  label = tf.strings.split(''.join([i for i in label[-1].numpy().decode() if not i.isdigit()]), sep='.jpg')

  return image,label[0]

###############################################################
## VARIABLES
###############################################################


## model inputs
data_path= os.getcwd()+os.sep+"data/nwpu/full/"+str(TARGET_SIZE)

test_samples_fig =  os.getcwd()+os.sep+'results/nwpu_sample_2class_mv2_model1_est24samples.png'

cm_filename =  os.getcwd()+os.sep+'results/nwpu_sample_2class_mv2_model1_cm_val.png'

sample_data_path=  os.getcwd()+os.sep+"data/nwpu/full/sample"

filepath =  os.getcwd()+os.sep+'results/nwpu_full_11class_mv2_best_weights_model1.h5'

hist_fig =  os.getcwd()+os.sep+'results/nwpu_sample_11class_mv2_model1.png'

json_file =  os.getcwd()+os.sep+'data/nwpu/nwpu_11classes.json'

patience = 10

with open(json_file) as f:
    class_dict = json.load(f)

# string names
CLASSES = [class_dict[k] for k in class_dict.keys()]

CLASSES = [c.encode() for c in CLASSES]
print(CLASSES)


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
val_ds = get_validation_dataset()
augmented_train_ds, augmented_val_ds = get_aug_datasets()



plt.figure(figsize=(12,12))
for imgs,lbls in train_ds.take(1):
  #print(lbls)
  for count,im in enumerate(imgs):
     plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
     plt.imshow(im)
     plt.title(CLASSES[lbls.numpy()[count]].decode(), fontsize=8)
     plt.axis('off')
# plt.show()
plt.savefig( os.getcwd()+os.sep+'results/nwpu_11class_trainsamples.png', dpi=200, bbox_inches='tight')

plt.figure(figsize=(12,12))
for imgs,lbls in val_ds.take(1):
  print(lbls)
  for count,im in enumerate(imgs):
     plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
     plt.imshow(im)
     plt.title(CLASSES[lbls.numpy()[count]].decode(), fontsize=8)
     plt.axis('off')
# plt.show()
plt.savefig( os.getcwd()+os.sep+'results/nwpu_11class_valsamples.png', dpi=200, bbox_inches='tight')


## data augmentation is typically used
## the primary purpose is regularization

plt.figure(figsize=(12,12))
for im,l in augmented_train_ds.take(1):
    for count,im in enumerate(im):
       plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
       plt.imshow(im)
       plt.title(CLASSES[l[count]].decode(), fontsize=8)
       plt.axis('off')
# plt.show()
plt.savefig( os.getcwd()+os.sep+'results/nwpu_11class_augtrainsamples.png', dpi=200, bbox_inches='tight')



###+===================================================
lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

model = transfer_learning_mobilenet_model(len(CLASSES), (TARGET_SIZE, TARGET_SIZE, 3), dropout_rate=0.5)

model.compile(optimizer=tf.keras.optimizers.Adam(),
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

# model.summary()

if do_train:
    history = model.fit(augmented_train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                          validation_data=augmented_val_ds, validation_steps=validation_steps,
                          callbacks=callbacks)

    # Plot training history
    plot_history(history, hist_fig)
    # plt.show()
    # plt.savefig(hist_fig, dpi=200, bbox_inches='tight')

    plt.close('all')
    K.clear_session()

else:
    model.load_weights(filepath)


##########################################################
### evaluate
val_ds = get_validation_eval_dataset()
loss, accuracy = model.evaluate(val_ds, batch_size=BATCH_SIZE)
print('Test Mean Accuracy: ', round((accuracy)*100, 2),' %')

##90%

##########################################################
### predict

sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))

plt.figure(figsize=(16,16))

for counter,f in enumerate(sample_filenames):
    image, im = file2tensor(f, 'mobilenet')
    plt.subplot(6,4,counter+1)
    name = sample_filenames[counter].split(os.sep)[-1].split('_')[0]
    name = ''.join([i for i in name if not i.isdigit()]).split('.jpg')[0]
    plt.title(name, fontsize=10)
    plt.imshow(tf.cast(image, tf.uint8))
    plt.axis('off')

    scores = model.predict(tf.expand_dims(im, 0) , batch_size=1)
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
val_ds = get_validation_eval_dataset()

labs, preds = get_label_pairs(val_ds, model)

# this time we invoke the optional 'thres', lower than defualt to see where probability leakage occurs
p_confmat(labs, preds, cm_filename, CLASSES, thres = 0.025)

#90%
