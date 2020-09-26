

#see mlmondays blog post:
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

##calcs
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

##plots
import matplotlib.pyplot as plt

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

###############################################################
## PLOTTING
###############################################################

#-----------------------------------
def plot_history(history, train_hist_fig):
    """
    plot_history(history, train_hist_fig)
    This function plots the training history of a model
    INPUTS:
        * history [dict]: the output dictionary of the model.fit() process, i.e. history = model.fit(...)
        * train_hist_fig [string]: the filename where the plot will be printed
    OPTIONAL INPUTS: None
    GLOBAL INPUTS: None
    OUTPUTS: None (figure printed to file)
    """
    n = len(history.history['loss'])

    plt.figure(figsize=(10,10))
    plt.subplot(121)
    plt.plot(np.arange(1,n+1), history.history['loss'], 'b', label='train loss')
    plt.plot(np.arange(1,n+1), history.history['val_loss'], 'k', label='validation loss')
    plt.xlabel('Epoch number', fontsize=10); plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=10)

    # plt.show()
    plt.savefig(train_hist_fig, dpi=200, bbox_inches='tight')



def visualize_detections(
    image, boxes, classes, scores, counter, str_prefix, figsize=(7, 7), linewidth=1, color=[0, 0, 1]
):
    """
    visualize_detections(image, boxes, classes, scores, counter, str_prefix, figsize=(7, 7), linewidth=1, color=[0, 0, 1])
    ""
    This function
    INPUTS:
        * val_dataset [tensorflow dataset]: validation dataset
        * train_dataset [tensorflow dataset]: training dataset
    OPTIONAL INPUTS: None
    OUTPUTS:
        * val_dataset [tensorflow dataset]: validation dataset
        * train_dataset [tensorflow dataset]: training dataset
    GLOBAL INPUTS: None
    """
    image = np.array(image, dtype=np.uint8)
    fig =plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.savefig(str_prefix+str(counter)+'.png', dpi=200, bbox_inches='tight')
    plt.close('all')
