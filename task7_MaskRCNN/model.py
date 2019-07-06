#!/usr/bin/env python
# Omdena Mask Regional Convolutional Neural Network (Mask R-CNN) for UN World Food Program
# Written by Erick Galinkin
# Note: I've opted to use Tensorflow 2.0 here. It should be easy to change if need be, but I'm assuming
# that support for TF2 will be longer-lived, and we get to play with all of the Keras abstractions without having to
# import a second library.
# This implementation was heavily inspired by Matterport's Mask RCNN and Tensorflow's expert CNN tutorial

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Input, Conv2D, BatchNormalization, Add, Activation, ZeroPadding2D,\
    MaxPool2D, Lambda, UpSampling2D, Layer
from tensorflow.python.keras import Model

data_dir = './images/'


# It may be prudent to rip this whole thing out and use Thomas's pre-trained Resnet-50
class Backbone(Model):
    """A ResNet50 implementation which serves as the backbone for the Mask R-CNN
    Has 2 static methods associated with it - conv_block and identity_block.
    We assume that we want to always train stage 5 of the network.
    Allows for a boolean train or freeze of batch norm layers.

    Due to the way batch norm works, we often want trainable to be None type, but may want to freeze it for smaller
    datasets.
    """
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv_1 = Conv2D(64, (7, 7), strides=(2, 2), use_bias=True)
        self.batchnorm = BatchNormalization(trainable=True)
        self.maxpool = MaxPool2D((3, 3), strides=(2, 2), padding="same")
        self.zero_pad = ZeroPadding2D((3,3))
        self.activation = Activation('relu')

    @staticmethod
    def conv_block(input_tensor, kernel_size, filters, use_bias=True, trainable=None):
        filter1, filter2, filter3 = filters
        x = Conv2D(filter1, (1, 1), strides=(2, 2), use_bias=use_bias)(input_tensor)
        x = BatchNormalization()(x, trainable=trainable)
        x = Activation('relu')(x)
        x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', use_bias=use_bias)(x)
        x = BatchNormalization()(x, trainable=trainable)
        x = Activation('relu')(x)
        x = Conv2D(filter3, (1, 1), use_bias=use_bias)(x)
        x = BatchNormalization()(x, trainable=trainable)
        shortcut = Conv2D(filter3, (1, 1), strides=(2, 2), use_bias=True)(input_tensor)
        shortcut = BatchNormalization()(shortcut, trainable=trainable)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    @staticmethod
    def identity_block(input_tensor, kernel_size, filters, use_bias=True, trainable=None):
        filter1, filter2, filter3 = filters
        x = Conv2D(filter1, (1, 1), use_bias=use_bias)(input_tensor)
        x = BatchNormalization()(x, trainable=trainable)
        x = Activation('relu')(x)
        x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', use_bias=True)(x)
        x = BatchNormalization()(x, trainable=trainable)
        x = Activation('relu')(x)
        x = Conv2D(filter3, (1, 1), use_bias=True)(x)
        x = BatchNormalization()(x, trainable=trainable)
        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def resnet_50(self, input_image, trainable=None):
        x = self.zero_pad(input_image)
        x = self.conv_1(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        C1 = x = self.maxpool(x)
        x = self.conv_block(x, 3, [64, 64, 256], trainable=trainable)
        x = self.identity_block(x, 3, [64, 64, 256], trainable=trainable)
        C2 = x = self.identity_block(x, 3, [64, 64, 256], trainable=trainable)
        x = self.conv_block(x, 3, [128, 128, 512], trainable=trainable)
        x = self.identity_block(x, 3, [128, 128, 512], trainable=trainable)
        x = self.identity_block(x, 3, [128, 128, 512], trainable=trainable)
        C3 = x = self.identity_block(x, 3, [128, 128, 512], trainable=trainable)
        x = self.conv_block(x, 3, [256, 256, 1024], trainable=trainable)
        # Per Matterport implementation, if we wanted to change this to resnet 101, we'd have 22 instead of 5 blocks
        x = self.identity_block(x, 3, [256, 256, 1024], trainable=trainable)
        x = self.identity_block(x, 3, [256, 256, 1024], trainable=trainable)
        x = self.identity_block(x, 3, [256, 256, 1024], trainable=trainable)
        x = self.identity_block(x, 3, [256, 256, 1024], trainable=trainable)
        x = self.identity_block(x, 3, [256, 256, 1024], trainable=trainable)
        C4 = x
        x = self.conv_block(x, 3, [512, 512, 2048], trainable=trainable)
        x = self.identity_block(x, 3, [512, 512, 2048], trainable=trainable)
        C5 = self.identity_block(x, 3, [512, 512, 2048], trainable=trainable)
        return [C1, C2, C3, C4, C5]


class MaskRCNN(Model):
    """Mask R-CNN implementation built on top of ResNet50"""
    def __init__(self, mode, config, model_dir):
        super(MaskRCNN, self).__init__()
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.model = self.build(mode=mode, config=config)

    def get_anchors(self, image_shape):
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            self.anchors = a
            self._anchor_cache[tuple]
        return self._anchor_cache[tuple(image_shape)]

    def build(self, mode, config):
        h, w = config.IMAGE_SHAPE[:2]
        if h/2**6 != int(h/2**6) or w/2**6 != int(w/2**6):
            raise Exception("Image size must be a multiple of 64 to allow up and downscaling")
        input_image = Input(shape=[None, None, config.IMAGE_SHAPE[2]])

        if mode == "training":
            input_rpn_match = Input(shape=[None, 1], dtype=tf.int32)
            input_rpn_bbox = Input(shape=[None, 4], dtype=tf.float32)

            # We're not *super* concerned with class id, but it's included.
            input_gt_class_id = Input(shape=[None], dtype=tf.int32)
            input_gt_boxes = Input(shape=[None, 4], dtype=tf.float32)

            gt_boxes = Lambda(lambda x: norm_boxes(x, tf.shape(input_image)[1:3]))(input_gt_boxes)
            input_gt_masks = Input(shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None], dtype=bool)

        elif mode == "inference":
            input_anchors = Input(shape=[None, 4])

        resnet = Backbone()
        _, C2, C3, C4, C5 = resnet.resnet50(input_image, trainable=config.TRAINABLE)

        P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1))(C5)
        P4 = Add()([UpSampling2D(size=(2, 2))(P5), Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1))(C4)])
        P3 = Add()([UpSampling2D(size=(2, 2))(P4), Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1))(C3)])
        P2 = Add()([UpSampling2D(size=(2, 2))(P3), Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1))(C2)])
        P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME")(P2)
        P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME")(P3)
        P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME")(P4)
        P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding="SAME")(P5)
        P6 = MaxPool2D(pool_size=(1, 1), strides=2)(P5)

        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mask_rcnn_feature_maps = [P2, P3, P4, P5]

        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE, ) + anchors.shape)
        else:
            anchors = input_anchors

        rpn = build_rpn(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        layer_outputs = []
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        output_names = ["rpn_class_logits", "rpn_class"]


##################################################################################################################
# The following classes and functions were copied directly from Matterport's Mask R-CNN implementation with only #
# very light modification from the original functionality (converting from Keras to TF 2.0) by Erick Galinkin.   #
# Copyright (c) 2017 Matterport, Inc.                                                                            #
# Licensed under the MIT License (see LICENSE for details)                                                       #
# Written by Waleed Abdulla                                                                                      #
##################################################################################################################

class ProposalLayer(Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.
    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates
    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]

    This layer only very gently modified from Matterport's implementation.
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.RPN_BBOX_STD_DEV, [1, 1, 4])
        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.config.PRE_NMS_LIMIT, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                    self.config.IMAGES_PER_GPU,
                                    names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = utils.batch_slice([pre_nms_anchors, deltas],
                                  lambda x, y: apply_box_deltas(x, y),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        window = np.array([0, 0, 1, 1], dtype=np.float32)
        boxes = utils.batch_slice(boxes,
                                  lambda x: clip_boxes(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # Non-max suppression
        def nms(boxes, scores):
            indices = tf.image.non_max_suppression(
                boxes, scores, self.proposal_count,
                self.nms_threshold, name="rpn_non_max_suppression")
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals
        proposals = utils.batch_slice([boxes, scores], nms,
                                      self.config.IMAGES_PER_GPU)
        return proposals

    def compute_output_shape(self, input_shape):
        return None, self.proposal_count, 4


class PyramidROIAlign(Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.
    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]
    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]
    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.

    This layer only gently modified from Matterport's implementation
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = parse_image_meta_graph(image_meta)['image_shape'][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = tf.math.log(tf.sqrt(h * w) / (224.0 / tf.sqrt(image_area)))/tf.math.log(2.0)  # TF has no log base2
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )


def norm_boxes(boxes, shape):
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)


def apply_box_deltas(boxes, deltas):
    # Convert the x and y coordinates to height and width
    width = boxes[:, 3] - boxes[:, 1]
    height = boxes[:, 2] - boxes[:, 0]
    center_x = boxes[:, 1] + (width / 2)
    center_y = boxes[:, 0] + (height/2)
    center_x += deltas[:, 1] * width
    center_y += deltas[:, 0] * height


def clip_boxes(boxes, window):
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.
    meta: [batch, meta length] where meta length depends on NUM_CLASSES
    Returns a dict of the parsed tensors.

    This function copied directly from Matterport's implementation to support the PyramidROIAlign layer
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        "image_id": image_id,
        "original_image_shape": original_image_shape,
        "image_shape": image_shape,
        "window": window,
        "scale": scale,
        "active_class_ids": active_class_ids,
    }

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.
    Inputs:
    proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.
    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))]
    masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped to bbox
           boundaries and resized to neural network output size.
    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                   name="trim_gt_class_ids")
    gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                         name="trim_gt_masks")

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
    crowd_boxes = tf.gather(gt_boxes, crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Compute overlaps with crowd boxes [proposals, crowd_boxes]
    crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
    crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
    no_crowd_bool = (crowd_iou_max < 0.001)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                         config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Assign positive ROIs to GT masks
    # Permute masks to [N, height, width, 1]
    transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)
    # Pick the right mask for each ROI
    roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

    # Compute mask targets
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # Transform ROI coordinates from normalized image space
        # to normalized mini-mask space.
        y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
        gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    box_ids = tf.range(0, tf.shape(roi_masks)[0])
    masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                     box_ids,
                                     config.MASK_SHAPE)
    # Remove the extra dimension from masks.
    masks = tf.squeeze(masks, axis=3)

    # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    # binary cross entropy loss.
    masks = tf.round(masks)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

    return rois, roi_gt_class_ids, deltas, masks


class DetectionTargetLayer(KE.Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.
    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type
    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox", "target_mask"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes, gt_masks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),  # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
            (None, self.config.TRAIN_ROIS_PER_IMAGE, self.config.MASK_SHAPE[0],
             self.config.MASK_SHAPE[1])  # masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]
##################################################################################################################


# There is no labeled data yet, so this is more or less just a mock-up. Once we have data, we'll read it in from a
# path and then return whatever format we need.
def ingest_data(path):
    pass
    return None


# Once again, once we know what format our data is in, we can fill this bit out. It will probably be some kind of
# wrapper around sklearn's train_test_split, but that's more or less an implementation detail.
def split_data(data):
    pass
    return None, None, None, None


def train_model(images, labels, epochs):
    model = MaskRCNN()
    # Probably want to use a different loss - but it's fine for now.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name='training_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    optimizer = tf.keras.optimizers.RMSprop()
    for epoch in range(epochs):
        with tf.GradientTape as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

        print('Epoch {}, Training Loss: {}, Training Accuracy {}'.format(
            epoch+1, train_loss.result(), train_accuracy.result()*100))

    return model


def validate_model(model, images, labels):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    val_loss = tf.keras.metrics.Mean(name='validation_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='validation_accuracy')

    predictions = model(images)
    loss = loss_object(labels, predictions)
    validation_loss = val_loss(loss)
    validation_accuracy = val_accuracy(labels, predictions)
    return validation_loss, validation_accuracy


if __name__ == "__main__":
    path_to_data = None  # TODO: Find the best way to populate this path
    data = ingest_data(path_to_data)
    X_train, X_val, y_train, y_val = split_data(data)
    model = train_model(X_train, y_train, EPOCHS)
    validation_loss, validation_accuracy = validate_model(model, X_val, y_val)
    print('Validation loss: {}, Validation accuracy: {}'.format(
        validation_loss.result(), validation_accuracy.result()*100))
