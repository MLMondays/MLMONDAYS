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
    return get_batched_dataset_oysternet(training_filenames)

def get_validation_dataset():
    """
    This function will return a batched dataset for model training
    INPUTS: None
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: validation_filenames
    OUTPUTS: batched data set object
    """
    return get_batched_dataset_oysternet(validation_filenames)


###############################################################
## VARIABLES
###############################################################

data_path= os.getcwd()+os.sep+"data/oysternet/"+str(TARGET_SIZE)

sample_data_path = os.getcwd()+os.sep+"data/oysternet/sample"

trainsamples_fig = os.getcwd()+os.sep+'results/oysternet_sample_2class_trainsamples.png'
valsamples_fig = os.getcwd()+os.sep+'results/oysternet_sample_2class_valsamples.png'

augsamples_fig = os.getcwd()+os.sep+'results/oysternet_sample_2class_augtrainsamples.png'

filepath =  os.getcwd()+os.sep+'results/oysternet_subset_2class_custom_best_weights_model768.h5'

hist_fig = os.getcwd()+os.sep+'results/oysternet_sample_2class_custom_model1.png'

test_samples_fig = os.getcwd()+os.sep+'results/oysternet_sample_2class_custom_model1_est24samples.png'

patience = 20

###############################################################
## EXECUTION
###############################################################

#-------------------------------------------------
training_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*train*.tfrec'))

validation_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*val*.tfrec'))

print('.....................................')
print('Reading files and making datasets ...')

num_files =  (len(training_filenames)+len(validation_filenames))

nb_images = ims_per_shard * num_files
print(nb_images)

validation_steps = int(nb_images // num_files * len(validation_filenames)) // BATCH_SIZE
steps_per_epoch = int(nb_images // num_files * len(training_filenames)) // BATCH_SIZE

print(steps_per_epoch)
print(validation_steps)

train_ds = get_training_dataset()
val_ds = get_validation_dataset()

print('.....................................')
print('Printing examples to file ...')


plt.figure(figsize=(16,16))
for imgs,lbls in train_ds.take(1):
  #print(lbls)
  for count,(im,lab) in enumerate(zip(imgs, lbls)):
     plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
     plt.imshow(im)
     plt.imshow(lab, cmap=plt.cm.gray, alpha=0.5)
     plt.axis('off')
# plt.show()
plt.savefig(trainsamples_fig, dpi=200, bbox_inches='tight')
plt.close('all')

del imgs, lbls, im, lab

plt.figure(figsize=(16,16))
for imgs,lbls in val_ds.take(1):
  #print(lbls)
  for count,(im,lab) in enumerate(zip(imgs, lbls)):
     plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
     plt.imshow(im)
     plt.imshow(lab, cmap=plt.cm.gray, alpha=0.5)
     plt.axis('off')
# plt.show()
plt.savefig(valsamples_fig, dpi=200, bbox_inches='tight')
plt.close('all')

del imgs, lbls, im, lab


# augmented_train_ds, augmented_val_ds = get_aug_datasets()
#
# plt.figure(figsize=(16,16))
# for im,l in augmented_train_ds.take(1):
#     for count,im in enumerate(im):
#        plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
#        plt.imshow(im)
#        plt.title(CLASSES[l[count]], fontsize=8)
#        plt.axis('off')
# # plt.show()
# plt.savefig(augsamples_fig, dpi=200, bbox_inches='tight')

print('.....................................')
print('Creating and compiling model ...')

nclass=1
model = res_unet((TARGET_SIZE, TARGET_SIZE, 3), BATCH_SIZE, 'binary', nclass)
# model.compile(optimizer = 'adam', loss = dice_coef_loss, metrics = [dice_coef])
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [mean_iou])

# model.summary()

earlystop = EarlyStopping(monitor="val_loss",
                              mode="min", patience=patience)

# set checkpoint file
model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)


lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lrfn(epoch), verbose=True)

callbacks = [model_checkpoint, earlystop, lr_callback]


do_train = False #True

if do_train:
    print('.....................................')
    print('Training model ...')
    history = model.fit(train_ds, steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
                          validation_data=val_ds, validation_steps=validation_steps,
                          callbacks=callbacks)

    # Plot training history
    plot_seg_history(history, hist_fig)

    plt.close('all')
    K.clear_session()

else:
    model.load_weights(filepath)


##########################################################
### evaluate
# loss, accuracy = model.evaluate(get_validation_eval_dataset("binary"), batch_size=BATCH_SIZE)
# print('Test Mean Accuracy: ', round((accuracy)*100, 2),' %')
print('.....................................')
print('Evaluating model ...')
scores = model.evaluate(val_ds, steps=validation_steps)

print('loss={loss:0.4f}, Mean Dice={dice_coef:0.4f}'.format(loss=scores[0], dice_coef=scores[1]))

##73%

##########################################################
### predict
print('.....................................')
print('Using model for prediction on jpeg images ...')

sample_filenames = sorted(tf.io.gfile.glob(sample_data_path+os.sep+'*.jpg'))

make_sample_seg_plot(model, sample_filenames, test_samples_fig)
