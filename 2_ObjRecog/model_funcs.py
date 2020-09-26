
#see mlmondays blog post:
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

##calcs
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from data_funcs import LabelEncoderCoco, convert_to_corners

SEED=42
np.random.seed(SEED)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

tf.random.set_seed(SEED)

###############################################################
## MODEL BUILDING
###############################################################

def compute_iou(boxes1, boxes2):
    """
    compute_iou(boxes1, boxes2)
    This function computes pairwise IOU matrix for given two sets of boxes
    INPUTS:
        * boxes1: A tensor with shape `(N, 4)` representing bounding boxes
          where each box is of the format `[x, y, width, height]`.
        * boxes2: A tensor with shape `(M, 4)` representing bounding boxes
          where each box is of the format `[x, y, width, height]`.
    OPTIONAL INPUTS: None
    OUTPUTS:
        *  pairwise IOU matrix with shape `(N, M)`, where the value at ith row
           jth column holds the IOU between ith box and jth box from
           boxes1 and boxes2 respectively.
    GLOBAL INPUTS: None
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


# ## Implementing Anchor generator
# Anchor boxes are fixed sized boxes that the model uses to predict the bounding
# box for an object. It does this by regressing the offset between the location
# of the object's center and the center of an anchor box, and then uses the width
# and height of the anchor box to predict a relative scale of the object. In the
# case of RetinaNet, each location on a given feature map has nine anchor boxes
# (at three scales and three ratios).

class AnchorBox:
    """
    "AnchorBox"
    ## Code from https://keras.io/examples/vision/retinanet/
    Generates anchor boxes.
    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.
    INPUTS:
      * aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      * scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      * num_anchors: The number of anchor boxes at each location on feature map
      * areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      * strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    OPTIONAL INPUTS: None
    OUTPUTS:
        * anchor boxes for all the feature maps, stacked as a single tensor with shape
        `(total_anchors, 4)`, when AnchorBox._get_anchors() is called
    GLOBAL INPUTS: None
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """
        "_get_anchors"
        ## Code from https://keras.io/examples/vision/retinanet/
        Generates anchor boxes for a given feature map size and level
        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.
        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """
        "get_anchors"
        ## Code from https://keras.io/examples/vision/retinanet/
        Generates anchor boxes for all the feature maps of the feature pyramid.
        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.
        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in range(3, 8)
        ]
        return tf.concat(anchors, axis=0)



def get_backbone():
    """
    get_backbone()
    ## Code from https://keras.io/examples/vision/retinanet/
    ""
    This function Builds ResNet50 with pre-trained imagenet weights
    INPUTS: None
    OPTIONAL INPUTS: None
    OUTPUTS:
        * keras Model
    GLOBAL INPUTS: BATCH_SIZE
    """
    backbone = tf.keras.applications.ResNet50(
        include_top=False, input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return tf.keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )


"""
## Building Feature Pyramid Network as a custom layer
"""


class FeaturePyramid(tf.keras.layers.Layer):
    """
    "FeaturePyramid"
    ## Code from https://keras.io/examples/vision/retinanet/
    This class builds the Feature Pyramid with the feature maps from the backbone.
    INPUTS:
      * num_classes: Number of classes in the dataset.
      * backbone: The backbone to build the feature pyramid from. Currently supports ResNet50 only (the output of get_backbone())
    OPTIONAL INPUTS: None
    OUTPUTS:
        * the 5-feature pyramids (feature maps) at strides `[8, 16, 32, 64, 128]`
    GLOBAL INPUTS: None
    """

    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = tf.keras.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = tf.keras.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = tf.keras.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = tf.keras.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = tf.keras.layers.UpSampling2D(2)

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output


def build_head(output_filters, bias_init):
    """
    "build_head"
    ## Code from https://keras.io/examples/vision/retinanet/
    This function builds the class/box predictions head.
    INPUTS:
        * output_filters: Number of convolution filters in the final layer.
        * bias_init: Bias Initializer for the final convolution layer.
    OPTIONAL INPUTS: None
    OUTPUTS:
        * a keras sequential model representing either the classification
          or the box regression head depending on `output_filters`.
    GLOBAL INPUTS: None
    """
    head = tf.keras.Sequential([tf.keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            tf.keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(tf.keras.layers.ReLU())
    head.add(
        tf.keras.layers.Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head


"""
## Building RetinaNet using a subclassed model
"""


class RetinaNet(tf.keras.Model):
    """
    "RetinaNet"
    ## Code from https://keras.io/examples/vision/retinanet/
    This class returns a subclassed Keras model implementing the RetinaNet architecture.
    INPUTS:
        * num_classes: Number of classes in the dataset.
        * backbone: The backbone to build the feature pyramid from. Supports ResNet50 only.
    OPTIONAL INPUTS: None
    OUTPUTS:
        * val_dataset [tensorflow dataset]: validation dataset
        * train_dataset [tensorflow dataset]: training dataset
    GLOBAL INPUTS: None
    """

    def __init__(self, num_classes, backbone=None, **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(
                tf.reshape(self.cls_head(feature), [N, -1, self.num_classes])
            )
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return tf.concat([box_outputs, cls_outputs], axis=-1)


"""
## Implementing a custom layer to decode predictions
"""


class DecodePredictions(tf.keras.layers.Layer):
    """
    "DecodePredictions"
    ## Code from https://keras.io/examples/vision/retinanet/
    This class creates a Keras layer that decodes predictions of the RetinaNet model.
    INPUTS:
        * num_classes: Number of classes in the dataset
        * confidence_threshold: Minimum class probability, below which detections
          are pruned.
        * nms_iou_threshold: IOU threshold for the NMS operation
        * max_detections_per_class: Maximum number of detections to retain per class.
        * max_detections: Maximum number of detections to retain across all classes.
        * box_variance: The scaling factors used to scale the bounding box predictions.
    OPTIONAL INPUTS: None
    OUTPUTS:
        * a keras layer to decode predictions
    GLOBAL INPUTS: None
    """

    def __init__(
        self,
        num_classes=80,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )


###############################################################
## MODEL TRAINING
###############################################################

class RetinaNetBoxLoss(tf.losses.Loss):
    """
    "RetinaNetBoxLoss"
    ## Code from https://keras.io/examples/vision/retinanet/
    This class implements smooth L1 loss
    INPUTS:
        * y_true [tensor]: label observations
        * y_pred [tensor]: label estimates
    OPTIONAL INPUTS: None
    OUTPUTS:
        * loss [tensor]
    GLOBAL INPUTS: None
    """

    def __init__(self, delta):
        super(RetinaNetBoxLoss, self).__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self._delta = delta

    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClassificationLoss(tf.losses.Loss):
    """
    "RetinaNetClassificationLoss"
    ## Code from https://keras.io/examples/vision/retinanet/
    This class implements Focal loss
    INPUTS:
        * y_true [tensor]: label observations
        * y_pred [tensor]: label estimates
    OPTIONAL INPUTS: None
    OUTPUTS:
        * loss [tensor]
    GLOBAL INPUTS: None
    """

    def __init__(self, alpha, gamma):
        super(RetinaNetClassificationLoss, self).__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetLoss(tf.losses.Loss):
    """
    "RetinaNetLoss"
    ## Code from https://keras.io/examples/vision/retinanet/
    This class is a wrapper to sum RetinaNetClassificationLoss and RetinaNetClassificationLoss outputs
    INPUTS:
        * y_true [tensor]: label observations
        * y_pred [tensor]: label estimates
    OPTIONAL INPUTS: None
    OUTPUTS:
        * loss [tensor]
    GLOBAL INPUTS: None
    """

    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0, delta=1.0):
        super(RetinaNetLoss, self).__init__(reduction="auto", name="RetinaNetLoss")
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[:, :, :4]
        box_predictions = y_pred[:, :, :4]
        cls_labels = tf.one_hot(
            tf.cast(y_true[:, :, 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[:, :, 4:]
        positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
        loss = clf_loss + box_loss
        return loss
