import tensorflow as tf

import requests, os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import random
from collections import defaultdict
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from collections import Counter

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
