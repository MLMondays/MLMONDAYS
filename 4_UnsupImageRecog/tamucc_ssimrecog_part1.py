
###############################################################
## IMPORTS
###############################################################
from imports import *

###############################################################
### DATA FUNCTIONS
###############################################################

#-----------------------------------
def get_training_dataset():
  return get_batched_dataset(training_filenames)

#-----------------------------------
def get_validation_dataset():
  return get_batched_dataset(validation_filenames)

def get_validation_eval_dataset():
  return get_eval_dataset(validation_filenames)

#-----------------------------------
def read_tfrecord(example):
    """
    This function
    """
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),   # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.uint8) #float32) / 255.0
    image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])

    class_label = tf.cast(example['class'], tf.int32)

    return image, class_label

#-----------------------------------
def get_batched_dataset(filenames):
    """
    This function
    """
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False  ##True?

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

    dataset = dataset.cache() # This dataset fits in RAM
    #dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(AUTO) #

    return dataset

###############################################################
### DATA FUNCTIONS
###############################################################

class AnchorPositivePairs(tf.keras.utils.Sequence):
    def __init__(self, num_batchs):
        self.num_batchs = num_batchs

    def __len__(self):
        return self.num_batchs

    def __getitem__(self, _idx):
        x = np.empty((2, num_classes, TARGET_SIZE, TARGET_SIZE, 3), dtype=np.float32)
        for class_idx in range(num_classes):
            examples_for_class = class_idx_to_train_idxs[class_idx]
            anchor_idx = random.choice(examples_for_class) # remove random dependency? replace with np.random?
            positive_idx = random.choice(examples_for_class) # remove random dependency? replace with np.random?
            while positive_idx == anchor_idx:
                positive_idx = random.choice(examples_for_class) # remove random dependency? replace with np.random?
            x[0, class_idx] = X_train[anchor_idx]
            x[1, class_idx] = X_train[positive_idx]
        return x

###############################################################
## VARIABLES
###############################################################

## model inputs
data_path= os.getcwd().replace('4_UnsupImageRecog', '1_ImageRecog')+os.sep+"data/tamucc/subset_2class/400"
test_samples_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_mv2_model1_est24samples.png'

cm_filename = os.getcwd()+os.sep+'results/tamucc_sample_2class_mv2_model1_cm_val.png'

sample_data_path= os.getcwd().replace('4_UnsupImageRecog', '1_ImageRecog')+os.sep+"data/tamucc/subset_2class/sample"

filepath = os.getcwd()+os.sep+'results/tamucc_subset_2class_custom_best_weights_model1.h5'

hist_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_custom_model1.png'

nn_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_custom_model1_acc_vs_nn.png'

trainsamples_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_trainsamples.png'

valsamples_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_validationsamples.png'

cm_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_cm_test.png'


CLASSES = [b'dev', b'undev']
patience = 10
num_embed_dim = 8
max_epochs = 100
lr = 1e-4

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
plt.figure(figsize=(16,16))
for imgs,lbls in train_ds.take(1):
  #print(lbls)
  for count,im in enumerate(imgs):
     plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
     plt.imshow(im)
     plt.title(CLASSES[lbls.numpy()[count]], fontsize=8)
     plt.axis('off')
# plt.show()
plt.savefig(trainsamples_fig, dpi=200, bbox_inches='tight')
plt.close('all')


val_ds = get_validation_dataset()
plt.figure(figsize=(16,16))
for imgs,lbls in val_ds.take(1):
  #print(lbls)
  for count,im in enumerate(imgs):
     plt.subplot(int(BATCH_SIZE/2),int(BATCH_SIZE/2),count+1)
     plt.imshow(im)
     plt.title(CLASSES[lbls.numpy()[count]], fontsize=8)
     plt.axis('off')
# plt.show()
plt.savefig(valsamples_fig, dpi=200, bbox_inches='tight')
plt.close('all')


training_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrec'))
nb_images = ims_per_shard * len(training_filenames)
print(nb_images)

num_batches = int(((1-VALIDATION_SPLIT) * nb_images) / BATCH_SIZE)
print(num_batches)

X_train = [] #np.zeros((nb_images,TARGET_SIZE, TARGET_SIZE, 3), dtype='uint8')
ytrain = []
train_ds = get_training_dataset()

counter = 0
for imgs,lbls in train_ds.take(num_batches):
  ytrain.append(lbls.numpy())
  for im in imgs:
    X_train.append(im)
    #X_train[counter] = im.numpy().astype('uint8')
    #counter += 1

X_train = np.array(X_train)
ytrain = np.hstack(ytrain)

X_test = [] #np.zeros((nb_images,TARGET_SIZE, TARGET_SIZE, 3), dtype='uint8')
ytest = []
test_ds = get_validation_dataset()

counter = 0
for imgs,lbls in test_ds.take(num_batches):
  ytest.append(lbls.numpy())
  for im in imgs:
    X_test.append(im)

X_test = np.array(X_test)
ytest = np.hstack(ytest)

# get X_train, y_train, X_test and y_test  arrays
X_train = X_train.astype("float32")
ytrain = np.squeeze(ytrain)
X_test = X_test.astype("float32")
ytest = np.squeeze(ytest)

# code repurposed from https://keras.io/examples/vision/metric_learning/
class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(ytrain):
    class_idx_to_train_idxs[y].append(y_train_idx)

class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(ytest):
    class_idx_to_test_idxs[y].append(y_test_idx)


num_classes = len(CLASSES)

model = get_embedding_model(TARGET_SIZE, num_classes, num_embed_dim)

num_batches = int(np.ceil(len(X_train) / len(CLASSES)))
print(num_batches)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
     metrics=['accuracy'],
)

earlystop = EarlyStopping(monitor="loss",
                              mode="min", patience=patience)

# set checkpoint file
model_checkpoint = ModelCheckpoint(filepath, monitor='loss',
                                verbose=0, save_best_only=True, mode='min',
                                save_weights_only = True)

callbacks = [model_checkpoint, earlystop]



history = model.fit(AnchorPositivePairs(num_batchs=num_batches), epochs=max_epochs, callbacks=callbacks)

plt.figure(figsize = (10,10))
plt.subplot(221)
plt.plot(history.history["loss"])
plt.xlabel('Model training epoch number')
plt.ylabel('Loss (soft cosine distance)')

plt.subplot(222)
plt.plot(history.history["accuracy"])
plt.xlabel('Model training epoch number')
plt.ylabel('Accuracy')
# plt.show()
plt.savefig(hist_fig, dpi=200, bbox_inches='tight')
plt.close('all')


gram_matrix = get_gram_matrix(X_test)

max_num_near_neighbours = 10
MN, confusion_matrix = conf_matrix(gram_matrix, max_num_near_neighbours, num_classes)


plt.plot(np.arange(1,max_num_near_neighbours,1), MN)
plt.ylabel('Average model accuracy')
plt.xlabel('Number of nearest neighbours')
# plt.show()
plt.savefig(nn_fig, dpi=200, bbox_inches='tight')
plt.close('all')


near_neighbours_per_example = np.argmax(MN)+1
print(near_neighbours_per_example)

near_neighbours = near_neighbours_from_samples(X_train, model)
confusion_matrix = get_cm_from_near_neighbours(ytrain, num_classes, near_neighbours, near_neighbours_per_example)


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
# Display a confusion matrix.
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[c.decode() for c in CLASSES])
disp.plot(include_values=True, cmap="viridis", ax=ax, xticks_rotation="vertical")
# plt.show()
plt.savefig(cm_fig.replace('test', 'train'), dpi=200, bbox_inches='tight')
plt.close('all')


# near_neighbours = near_neighbours_from_samples(X_test, model)
# confusion_matrix = get_cm_from_near_neighbours(ytest, num_classes, near_neighbours, near_neighbours_per_example)
#
#
# embeddings = model.predict(X_test)
#
# embeddings = tf.nn.l2_normalize(embeddings, axis=-1)
#
# gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
# near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]
#
#
# confusion_matrix = np.zeros((num_classes, num_classes))
#
# # For each class.
# for class_idx in range(num_classes):
#     # Consider "near_neighbours_per_example examples.
#     example_idxs = class_idx_to_test_idxs[class_idx][:near_neighbours_per_example]
#     for y_test_idx in example_idxs:
#         # And count the classes of its near neighbours.
#         for nn_idx in near_neighbours[y_test_idx][:-1]:
#             nn_class_idx = y_test[nn_idx]
#             confusion_matrix[class_idx, nn_class_idx] += 1
# confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
#
# fig = plt.figure(figsize=(15,15))
# ax = fig.add_subplot(111)
# # Display a confusion matrix.
# disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
# disp.plot(include_values=True, cmap="viridis", ax=ax, xticks_rotation="vertical")
# # plt.show()
#
# print("Mean accuracy: %f" % (np.mean(np.diag(confusion_matrix))))

min_neighbours = 1
max_neighbours = 5

CM = []

for near_neighbours_per_example in np.arange(min_neighbours,max_neighbours,1):

  near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]
  confusion_matrix = np.zeros((num_classes, num_classes))

  # For each class.
  for class_idx in range(num_classes):
      # Consider 10 examples.
      example_idxs = class_idx_to_test_idxs[class_idx][:near_neighbours_per_example] #[:10]
      for y_test_idx in example_idxs:
          # And count the classes of its near neighbours.
          for nn_idx in near_neighbours[y_test_idx][:-1]:
              #print("dist: %f, class: %i" % (gram_matrix[nn_idx, nn_idx], y_test[nn_idx]))
              nn_class_idx = y_test[nn_idx]
              confusion_matrix[class_idx, nn_class_idx] += 1
  CM.append(confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis])


cm = np.mean(CM, axis=0)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
# Display a confusion matrix.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(include_values=True, cmap="viridis", ax=ax, xticks_rotation="vertical")
plt.title("Ensemble confusion matrix")

print("Mean accuracy: %f" % (np.mean(np.diag(cm))))

# plt.show()
plt.savefig(cm_fig.replace('test', 'test_ensemble'), dpi=200, bbox_inches='tight')
plt.close('all')


cmv2 = np.std(CM, axis=0) / np.mean(CM, axis=0)

plt.figure(figsize=(10,10))
ax = plt.subplot(111)
ax.plot(100*np.diag(cmv2), 'k-o')
ax.set_xticks(np.arange(len(classes)))
ax.set_xticklabels(classes, fontsize=10, rotation=30)
plt.ylabel('Variability Index, %')

# plt.show()
plt.savefig(cm_fig.replace('test', 'test_ensemble_variability'), dpi=200, bbox_inches='tight')
plt.close('all')


# index in x_test
k =1000

near_neighbours_per_example = 3

plt.imshow(x_test[k])
plt.title(classes[y_test[k]])

e = model.predict(np.expand_dims(x_test[k],0))

near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]



class_idx = int(np.where(np.array(classes) == classes[y_test[k]])[0])

example_idxs = class_idx_to_test_idxs[class_idx][:near_neighbours_per_example]

#print(example_idxs)

N = []
for y_test_idx in example_idxs:
   for nn_idx in near_neighbours[y_test_idx][:-1]:
      nn_class_idx = y_test[nn_idx]
      N.append(nn_class_idx)

plt.figure(figsize=(10,10))
ax = plt.subplot(111)


score = 100*(np.bincount(N, minlength=len(classes))/len(N))

ind = np.argsort(score)

ax.plot(score[ind], 'k-o')
ax.set_xticks(np.arange(len(classes)))
ax.set_xticklabels(np.array(classes)[ind], fontsize=10, rotation=30)
plt.ylabel('Likelihood, %')

N = []
for near_neighbours_per_example in np.arange(min_neighbours,max_neighbours,1):
   near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]

   class_idx = int(np.where(np.array(classes) == classes[y_test[k]])[0])

   example_idxs = class_idx_to_test_idxs[class_idx][:near_neighbours_per_example]

   for y_test_idx in example_idxs:
      for nn_idx in near_neighbours[y_test_idx][:-1]:
         nn_class_idx = y_test[nn_idx]
         N.append(nn_class_idx)




plt.figure(figsize=(10,10))
ax = plt.subplot(111)

score = 100*(np.bincount(N, minlength=len(classes))/len(N))

ind = np.argsort(score)

ax.plot(score[ind], 'k-o')
ax.set_xticks(np.arange(len(classes)))
ax.set_xticklabels(np.array(classes)[ind], fontsize=10, rotation=30)
plt.ylabel('Likelihood, %')


from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

embeddings = model.predict(x_train)

embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

tl=TSNE(n_components=2, metric='cosine')
embedding_tsne=tl.fit_transform(embeddings)


embedding_tsne.shape


kmeans = KMeans(init='k-means++', n_clusters=num_classes, n_init=10)
kmeans.fit(embedding_tsne)

## adapted from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = embedding_tsne[:, 0].min() - 1, embedding_tsne[:, 0].max() + 1
y_min, y_max = embedding_tsne[:, 1].min() - 1, embedding_tsne[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)


# Put the result into a color plot
plt.figure(figsize=(15,15))
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.cividis, #plt.cm.Paired,
           aspect='auto', origin='lower')

# plot TSNE embeddings
plt.plot(embedding_tsne[:, 0], embedding_tsne[:, 1], 'k.', markersize=2)

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the TSNE-reduced data\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

for k in range(num_classes):
   ind = np.where(kmeans.labels_ == k)[0]
   plt.text(np.mean(embedding_tsne[ind, 0]), np.mean(embedding_tsne[ind, 1]), \
            classes[k] , color='r', fontsize=16)


embeddings_test = model.predict(x_test)

embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

tl=TSNE(n_components=2,  metric='cosine')
embedding_tsne_test = tl.fit_transform(embeddings_test)


## adapted from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = embedding_tsne_test[:, 0].min() - 1, embedding_tsne_test[:, 0].max() + 1
y_min, y_max = embedding_tsne_test[:, 1].min() - 1, embedding_tsne_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)



# Put the result into a color plot
plt.figure(figsize=(15,15))
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.cividis, #plt.cm.Paired,
           aspect='auto', origin='lower')

# plot TSNE embeddings
plt.plot(embedding_tsne_test[:, 0], embedding_tsne_test[:, 1], 'k.', markersize=2)

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the TSNE-reduced data\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

for k in range(num_classes):
   ind = np.where(kmeans.labels_ == k)[0]
   plt.text(np.mean(embedding_tsne_test[ind, 0]), np.mean(embedding_tsne_test[ind, 1]), \
            classes[k] , color='r', fontsize=16)
