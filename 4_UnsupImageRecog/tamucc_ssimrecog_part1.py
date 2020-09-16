
#see mlmondays blog post:
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"


import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from collections import Counter


SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


#-----------------------------------
def get_training_dataset():
  return get_batched_dataset(training_filenames)

#-----------------------------------
def get_validation_dataset():
  return get_batched_dataset(validation_filenames)

def get_validation_eval_dataset():
  return get_eval_dataset(validation_filenames)

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
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [TARGET_SIZE,TARGET_SIZE, 3])

    class_label = tf.cast(example['class'], tf.int32)

    return image, class_label


###############################################################
## VARIABLES
###############################################################

#what's the goal? Yo're trying to maximize your validation accuracy. You want your training accuracy to be high too, but not if the validation accuracy isn't as or nearly as high

## model inputs
data_path= os.getcwd().replace('4_UnsupImageRecog', '1_ImageRecog')+os.sep+"data/tamucc/subset_2class/400"

test_samples_fig = os.getcwd().replace('4_UnsupImageRecog', '1_ImageRecog')+os.sep+'results/tamucc_sample_2class_mv2_model1_est24samples.png'

cm_filename = os.getcwd().replace('4_UnsupImageRecog', '1_ImageRecog')+os.sep+'results/tamucc_sample_2class_mv2_model1_cm_val.png'

sample_data_path= os.getcwd().replace('4_UnsupImageRecog', '1_ImageRecog')+os.sep+"data/tamucc/subset_2class/sample"

filepath = os.getcwd().replace('4_UnsupImageRecog', '1_ImageRecog')+os.sep+'results/tamucc_subset_2class_custom_best_weights_model1.h5'

hist_fig = os.getcwd().replace('4_UnsupImageRecog', '1_ImageRecog')+os.sep+'results/tamucc_sample_2class_custom_model1.png'


trainsamples_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_trainsamples.png'
valsamples_fig = os.getcwd()+os.sep+'results/tamucc_sample_2class_validationsamples.png'

CLASSES = [b'dev', b'undev']
patience = 10

TARGET_SIZE = 400
VALIDATION_SPLIT = 0.4
ims_per_shard = 200
BATCH_SIZE = 6

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


training_filenames = sorted(tf.io.gfile.glob(data_path+os.sep+'*.tfrec'))
nb_images = ims_per_shard * len(training_filenames)
print(nb_images)

num_batches = int(((1-VALIDATION_SPLIT) * nb_images) / BATCH_SIZE)
print(num_batches)

num_batches = 10

X_train = []
ytrain = []
train_ds = get_training_dataset()
counter=0
for imgs,lbls in train_ds.take(num_batches):
  print(counter)
  counter += 1
  ytrain.append(lbls.numpy())
  for im in imgs:
    X_train.append(im)

X_train = np.array(X_train)

ytrain = np.hstack(ytrain)



# get x_train, y_train, x_test and y_test  arrays

x_train = x_train.astype("float32")
y_train = np.squeeze(y_train)
x_test = x_test.astype("float32")
y_test = np.squeeze(y_test)

# code repurposed from https://keras.io/examples/vision/metric_learning/

class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(y_train):
    class_idx_to_train_idxs[y].append(y_train_idx)

class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)


    num_classes = len(classes)

    # code repurposed from https://keras.io/examples/vision/metric_learning/

    class AnchorPositivePairs(keras.utils.Sequence):
        def __init__(self, num_batchs):
            self.num_batchs = num_batchs

        def __len__(self):
            return self.num_batchs

        def __getitem__(self, _idx):
            x = np.empty((2, num_classes, height_width, height_width, 3), dtype=np.float32)
            for class_idx in range(num_classes):
                examples_for_class = class_idx_to_train_idxs[class_idx]
                anchor_idx = random.choice(examples_for_class)
                positive_idx = random.choice(examples_for_class)
                while positive_idx == anchor_idx:
                    positive_idx = random.choice(examples_for_class)
                x[0, class_idx] = x_train[anchor_idx]
                x[1, class_idx] = x_train[positive_idx]
            return x


class EmbeddingModel(keras.Model):
    def train_step(self, data):
        # Note: Workaround for open issue, to be removed.
        if isinstance(data, tuple):
            data = data[0]
        anchors, positives = data[0], data[1]

        with tf.GradientTape() as tape:
            # Run both anchors and positives through model.
            anchor_embeddings = self(anchors, training=True)
            positive_embeddings = self(positives, training=True)

            # Calculate cosine similarity between anchors and positives. As they have
            # been normalised this is just the pair wise dot products.
            similarities = tf.einsum(
                "ae,pe->ap", anchor_embeddings, positive_embeddings
            )

            # Since we intend to use these as logits we scale them by a temperature.
            # This value would normally be chosen as a hyper parameter.
            temperature = 0.3 # 0.2
            similarities /= temperature

            # We use these similarities as logits for a softmax. The labels for
            # this call are just the sequence [0, 1, 2, ..., num_classes] since we
            # want the main diagonal values, which correspond to the anchor/positive
            # pairs, to be high. This loss will move embeddings for the
            # anchor/positive pairs together and move all other pairs apart.
            sparse_labels = tf.range(num_classes)
            loss = self.compiled_loss(sparse_labels, similarities)

        # Calculate gradients and apply via optimizer.
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (specifically the one for the loss value).
        self.compiled_metrics.update_state(sparse_labels, similarities)
        return {m.name: m.result() for m in self.metrics}



inputs = layers.Input(shape=(height_width, height_width, 3))
x = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu")(inputs) #
x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu")(x) #32
x = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu")(x) #64
x = layers.GlobalAveragePooling2D()(x)
embeddings = layers.Dense(units = num_embed_dim, activation=None)(x)
#embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

model = EmbeddingModel(inputs, embeddings)

max_epochs = 200
lr = 1e-4

num_batches = int(np.ceil(len(x_train) / len(classes)))
print(num_batches)


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
     metrics=['accuracy'],
)

history = model.fit(AnchorPositivePairs(num_batchs=num_batches), epochs=max_epochs)

plt.figure(figsize = (10,10))
plt.plot(history.history["loss"])
plt.xlabel('Model training epoch number')
plt.ylabel('Loss (soft cosine distance)')
plt.show()


# code from https://keras.io/examples/vision/metric_learning/

embeddings = model.predict(x_test)

embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)


max_num_near_neighbours = 20

# code adapted from https://keras.io/examples/vision/metric_learning/

MN = []; MJ = []
for near_neighbours_per_example in np.arange(1,max_num_near_neighbours,1):

  near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]

  confusion_matrix = np.zeros((num_classes, num_classes))

  # For each class.
  for class_idx in range(num_classes):
      # Consider 'near_neighbours' examples.
      example_idxs = class_idx_to_test_idxs[class_idx][:near_neighbours_per_example]
      for y_test_idx in example_idxs:
          # And count the classes of its near neighbours.
          for nn_idx in near_neighbours[y_test_idx][:-1]:
              nn_class_idx = y_test[nn_idx]
              #tally that class pairing
              confusion_matrix[class_idx, nn_class_idx] += 1

  # normalize by row totals to make the matrix stochastic
  confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

  # mean recall as the mean of diagonal elements
  MN.append(np.mean(np.diag(confusion_matrix)))
  # maximum recall
  MJ.append(np.max(np.diag(confusion_matrix)))


plt.plot(np.arange(1,max_num_near_neighbours,1), MN)
plt.plot(np.arange(1,max_num_near_neighbours,1), MJ)
plt.ylabel('accuracy')
plt.xlabel('number of nearest neighbours')

near_neighbours_per_example = np.argmax(MN)+1
print(near_neighbours_per_example)

embeddings = model.predict(x_train)

embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]

confusion_matrix = np.zeros((num_classes, num_classes))

# For each class.
for class_idx in range(num_classes):
    # Consider "near_neighbours_per_example" examples.
    example_idxs = class_idx_to_train_idxs[class_idx][:near_neighbours_per_example]
    for y_train_idx in example_idxs:
        # And count the classes of its near neighbours.
        for nn_idx in near_neighbours[y_train_idx][:-1]:
            nn_class_idx = y_train[nn_idx]
            confusion_matrix[class_idx, nn_class_idx] += 1
confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
# Display a confusion matrix.
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
disp.plot(include_values=True, cmap="viridis", ax=ax, xticks_rotation="vertical")
# plt.show()


embeddings = model.predict(x_test)

embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]


confusion_matrix = np.zeros((num_classes, num_classes))

# For each class.
for class_idx in range(num_classes):
    # Consider "near_neighbours_per_example examples.
    example_idxs = class_idx_to_test_idxs[class_idx][:near_neighbours_per_example]
    for y_test_idx in example_idxs:
        # And count the classes of its near neighbours.
        for nn_idx in near_neighbours[y_test_idx][:-1]:
            nn_class_idx = y_test[nn_idx]
            confusion_matrix[class_idx, nn_class_idx] += 1
confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
# Display a confusion matrix.
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
disp.plot(include_values=True, cmap="viridis", ax=ax, xticks_rotation="vertical")
# plt.show()

print("Mean accuracy: %f" % (np.mean(np.diag(confusion_matrix))))
