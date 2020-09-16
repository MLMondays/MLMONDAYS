
TARGET_SIZE = 400
VALIDATION_SPLIT = 0.4
ims_per_shard = 200
BATCH_SIZE = 6

num_classes = 2

#see mlmondays blog post:
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

##calcs
import tensorflow as tf
import numpy as np

##plots
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

##utils
from collections import defaultdict
from PIL import Image
from collections import Counter
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import random ## candidate for removel


SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print('GPU name: ', tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


###############################################################
### MODEL FUNCTIONS
###############################################################

# code repurposed from https://keras.io/examples/vision/metric_learning/

class EmbeddingModel(tf.keras.Model):
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


def get_gram_matrix(X_test):
    embeddings = model.predict(X_test)

    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
    return gram_matrix

# learning rate function
def lrfn(epoch):
    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        if epoch < rampup_epochs:
            lr = (max_lr - start_lr)/rampup_epochs * epoch + start_lr
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        else:
            lr = (max_lr - min_lr) * exp_decay**(epoch-rampup_epochs-sustain_epochs) + min_lr
        return lr
    return lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)


def near_neighbours_from_samples(X_train, model):
    embeddings = model.predict(X_train)

    embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    gram_matrix = np.einsum("ae,be->ab", embeddings, embeddings)
    near_neighbours = np.argsort(gram_matrix.T)[:, -(near_neighbours_per_example + 1) :]
    return near_neighbours

def get_cm_from_near_neighbours(ytrain, num_classes, near_neighbours, near_neighbours_per_example):

    confusion_matrix = np.zeros((num_classes, num_classes))

    # For each class.
    for class_idx in range(num_classes):
        # Consider "near_neighbours_per_example" examples.
        example_idxs = class_idx_to_train_idxs[class_idx][:near_neighbours_per_example]
        for y_train_idx in example_idxs:
            # And count the classes of its near neighbours.
            for nn_idx in near_neighbours[y_train_idx][:-1]:
                nn_class_idx = ytrain[nn_idx]
                confusion_matrix[class_idx, nn_class_idx] += 1
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    return confusion_matrix


def get_embedding_model(TARGET_SIZE, num_classes, num_embed_dim):

    inputs = tf.keras.layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation="relu")(inputs) #
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation="relu")(x) #32
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, activation="relu")(x) #64
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    embeddings = tf.keras.layers.Dense(units = num_embed_dim, activation=None)(x)
    #embeddings = tf.nn.l2_normalize(embeddings, axis=-1)

    model = EmbeddingModel(inputs, embeddings)
    return model

###############################################################
### PLOT FUNCTIONS
###############################################################

def conf_matrix(gram_matrix, max_num_near_neighbours, num_classes):
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
                  nn_class_idx = ytest[nn_idx]
                  #tally that class pairing
                  confusion_matrix[class_idx, nn_class_idx] += 1

      # normalize by row totals to make the matrix stochastic
      confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

      # mean recall as the mean of diagonal elements
      MN.append(np.mean(np.diag(confusion_matrix)))
      # maximum recall
      #MJ.append(np.max(np.diag(confusion_matrix)))
    return MN, confusion_matrix
