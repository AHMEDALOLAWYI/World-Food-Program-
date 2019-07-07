#!/usr/bin/env python
# Omdena Mask Regional Convolutional Neural Network (Mask R-CNN) for UN World Food Program
# Written by Erick Galinkin
# Note: I've opted to use Tensorflow 2.0 here. It should be easy to change if need be, but I'm assuming
# that support for TF2 will be longer-lived, and we get to play with all of the Keras abstractions without having to
# import a second library.
# This implementation leans on Matterport's Mask RCNN and Tensorflow's expert CNN tutorial

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Input, Conv2D, BatchNormalization, Add, Activation, ZeroPadding2D,\
    MaxPool2D, Lambda, UpSampling2D
from tensorflow.python.keras import Model
import matterport_utils
from matterport_utils import DetectionLayer, DetectionTargetLayer, ProposalLayer, PyramidROIAlign
import os

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
            a = matterport_utils.generate_pyramid_anchors(
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
        input_image_meta = Input(shape=[config.IMAGE_META_SIZE])

        if mode == "training":
            input_rpn_match = Input(shape=[None, 1], dtype=tf.int32)
            input_rpn_bbox = Input(shape=[None, 4], dtype=tf.float32)

            # We're not *super* concerned with class id, but it's included.
            input_gt_class_ids = Input(shape=[None], dtype=tf.int32)
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
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bounding_box"]
        outputs = list(zip(*layer_outputs))
        outputs = [tf.python.keras.layers.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bounding_box = outputs

        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(proposal_count=proposal_count, nms_threshold=config.RPN_NMS_THRESHOLD, config=config)\
            ([rpn_class, rpn_bounding_box, anchors])

        if mode == "training":

            active_class_ids = Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)
            target_rois = rpn_rois

            # noinspection PyUnboundLocalVariable
            rois, target_class_ids, target_bounding_box, target_mask = DetectionTargetLayer(config)\
                ([target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            mask_rcnn_class_logits, mask_rcnn_class, mask_rcnn_bounding_box = \
                fpn_classifier(rois, mask_rcnn_feature_maps, input_image_meta, config.POOL_SIZE, config.NUM_CLASSES,
                               trainable=config.TRAINABLE, fc_layers_size=config.FPN_FC_LAYERS_SIZE)

            mask_rcnn_mask = build_fpn_mask(rois, mask_rcnn_feature_maps, input_image_meta, config.MASK_POOL_SIZE,
                                            config.NUM_CLASSES, trainable=config.TRAINABLE)

            output_rois = tf.identity(rois)

            # noinspection PyUnboundLocalVariable
            rpn_class_loss = Lambda(lambda x: calculate_rpn_class_loss(*x))([input_rpn_match, rpn_class_logits])
            rpn_bounding_loss = Lambda(lambda x: calculate_rpn_bounding_loss(config, *x))\
                ([input_rpn_bbox, input_rpn_match, rpn_bounding_box])
            class_loss = Lambda(lambda x: mask_rcnn_class_loss(*x))\
                ([target_class_ids, mask_rcnn_class_logits, active_class_ids])
            bounding_loss = Lambda(lambda x: mask_rcnn_bounding_loss(*x))\
                ([target_bounding_box, target_class_ids, mask_rcnn_bounding_box])
            mask_loss = Lambda(lambda x: mask_rcnn_mask_loss(*x))([target_mask, target_class_ids, mask_rcnn_mask])

            inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox, input_gt_class_ids,
                      input_gt_boxes, input_gt_masks]

            outputs = [rpn_class_logits, rpn_class, rpn_bounding_box, mask_rcnn_class_logits, mask_rcnn_class,
                       mask_rcnn_bounding_box, mask_rcnn_mask, rpn_rois, output_rois, rpn_class_loss, rpn_bounding_loss,
                       class_loss, bounding_loss, mask_loss]
            model = Model(inputs, outputs)

        else:
            mask_rcnn_class_logits, mask_rcnn_class, mask_rcnn_bounding_box = \
                fpn_classifier(rpn_rois, mask_rcnn_feature_maps, input_image_meta, config.POOL_SIZE, config.NUM_CLASSES,
                               trainable=config.TRAINABLE, fc_layers_size=config.FPN_FC_LAYERS_SIZE)
            detections = DetectionLayer(config)([rpn_rois, mask_rcnn_class, mask_rcnn_bounding_box, input_image_meta])
            detection_boxes = Lambda(lambda x: x[..., :4])(detections)
            mask_rcnn_mask = build_fpn_mask(detection_boxes, mask_rcnn_feature_maps, input_image_meta,
                                            config.MASK_POOL_SIZE, config.NUM_CLASSES, trainable=config.TRAINABLE)
            model = Model([input_image, input_image_meta, input_anchors], [detections, mask_rcnn_class,
                                                                           mask_rcnn_bounding_box, mask_rcnn_mask,
                                                                           rpn_rois, rpn_class, rpn_bounding_box])

        return model

    def find_last_checkpoint(self):
        key = self.config.NAME.lower()
        # Pick the last directory in the list of model directories
        directories = next(os.walk(self.model_dir))[1]
        directories = sorted(filter(lambda f: f.startswith(key), directories))
        dir_name = os.path.join(self.model_dir, directories[-1])
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = sorted(filter(lambda f: f.startswith("mask_rcnn"), checkpoints))
        # Return the last checkpoint in the last directory
        return os.path.join(dir_name, checkpoints[-1])






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
