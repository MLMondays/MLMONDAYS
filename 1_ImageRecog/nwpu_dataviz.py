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
### FUNCTIONS
###############################################################

#-----------------------------------
def get_training_dataset():
    """
    This function will return a batched dataset for training
    """
    return get_batched_dataset_viz(training_filenames)

def get_validation_dataset():
    """
    This function will return a batched dataset for validation
    """
    return get_batched_dataset_viz(validation_filenames)

#-----------------------------------
def get_batched_dataset_viz(filenames):
    """
    This function defines the flow of data from the tfrecord files,
    for visuzliation purposes (no .repeat)
    """
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = True

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    dataset = dataset.map(read_tfrecord_viz, num_parallel_calls=AUTO)

    dataset = dataset.cache() # This dataset fits in RAM
    # dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(AUTO) #

    return dataset

#-----------------------------------
def read_tfrecord_viz(example):
    """
    This function reads an example record from a tfrecord and decodes the image into a jpeg
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.uint8)
    image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])

    class_label = tf.cast(example['class'], tf.int32)

    return image, class_label

####================================================

data_path= os.getcwd()+os.sep+"data/nwpu/full/224"
json_file = os.getcwd()+os.sep+'data/nwpu/nwpu_11classes.json'


training_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrec'))

# CLASSES = ['undev', 'dev']

CLASSES = read_classes_from_json(json_file)
print(CLASSES)

CLASSES = [c.decode() for c in CLASSES]


nb_images = ims_per_shard * len(training_filenames)
print(nb_images)


num_batches = int(((1-VALIDATION_SPLIT) * nb_images) / BATCH_SIZE)
print(num_batches)
X_train = []
ytrain = []
train_ds = get_training_dataset()
for imgs,lbls in train_ds.take(num_batches):
  n = np.bincount(lbls, minlength=len(CLASSES))
  ytrain.append(lbls.numpy())
  for im in imgs:
    X_train.append(im)


X_train = np.array(X_train)

ytrain = np.hstack(ytrain)


# show examples per class

bs = 6
for class_idx in range(len(CLASSES)): # [0,1,2]:
  #show_one_class(class_idx=class_idx, bs=64)
  locs = np.where(ytrain == class_idx)
  samples = locs[:][0]
  #random.shuffle(samples)
  samples = samples[:bs]
  print("Total number of {} (s) in the dataset: {}".format(CLASSES[class_idx], len(locs[:][0])))
  X_subset = X_train[samples]
  plot_one_class(X_subset, samples, class_idx, bs, CLASSES, rows=3, cols=2)
  plt.savefig( os.getcwd()+os.sep+'results/nwpu_sample_11class_samples_'+CLASSES[class_idx]+'.png', dpi=200, bbox_inches='tight')
  plt.close('all')



# plot mean images per class

plot_mean_images(X_train, ytrain, CLASSES, rows=4, cols=3)
plt.savefig( os.getcwd()+os.sep+'results/nwpu_sample_11class_mean.png', dpi=200, bbox_inches='tight')
plt.close('all')


#### plot histograms


plot_distribution(X_train, ytrain, 0, CLASSES)
plt.savefig( os.getcwd()+os.sep+'results/nwpu_sample_11class_hist_'+CLASSES[0]+'.png', dpi=200, bbox_inches='tight')
plt.close('all')

plot_distribution(X_train, ytrain, 1, CLASSES)
plt.savefig( os.getcwd()+os.sep+'results/nwpu_sample_11class_hist_'+CLASSES[1]+'.png', dpi=200, bbox_inches='tight')
plt.close('all')

plot_distribution(X_train, ytrain, 2, CLASSES)
plt.savefig( os.getcwd()+os.sep+'results/nwpu_sample_11class_hist_'+CLASSES[2]+'.png', dpi=200, bbox_inches='tight')
plt.close('all')



num_samples = 500
X_subset = X_train[:num_samples]
X_subset = X_subset.reshape(num_samples,-1)
y_subset = ytrain[:num_samples]



num_components=100
pca = PCA(n_components=num_components)
reduced = pca.fit_transform(X_subset)
print('Cumulative variance explained by {} principal components: {}'.format(num_components, np.sum(pca.explained_variance_ratio_)))



# Create animation
tsne = TSNE(n_components=3,n_jobs=-1)
tsne_result = tsne.fit_transform(reduced)
tsne_result_scaled = StandardScaler().fit_transform(tsne_result)


fig, ax = plot_tsne(tsne_result_scaled,y_subset, CLASSES)
plt.savefig( os.getcwd()+os.sep+'results/nwpu_sample_11class_tsne_sample.png', dpi=200, bbox_inches='tight')
plt.close('all')


X_subset = X_train[:num_samples]
y_subset = ytrain[:num_samples]


f = visualize_scatter_with_images(tsne_result_scaled, y_subset,
                                  images = [np.reshape(i, (TARGET_SIZE,TARGET_SIZE,3)) for i in X_subset],
                                  image_zoom=0.5, xlim = (-2,2), ylim=(-2,2))

plt.savefig( os.getcwd()+os.sep+'results/nwpu_sample_11class_tsne_vizimages_sample.png', dpi=200, bbox_inches='tight')
plt.close('all')
