#!/usr/bin/env python
# Omdena Mask Regional Convolutional Neural Network (Mask R-CNN) for UN World Food Program
# Written by Erick Galinkin
# Note: I've opted to use Tensorflow 2.0 here. It should be easy to change if need be, but I'm assuming
# that support for TF2 will be longer-lived, and we get to play with all of the Keras abstractions without having to
# import a second library.
# This implementation leans on Matterport's Mask RCNN and Tensorflow's expert CNN tutorial

import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, Activation, ZeroPadding2D,\
    MaxPool2D, Lambda, UpSampling2D, TimeDistributed, Dense, Reshape, Conv2DTranspose
from tensorflow.keras import Model
import matterport_utils as mp_utils
from matterport_utils import DetectionLayer, DetectionTargetLayer, ProposalLayer, PyramidROIAlign
import os
import datetime

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
        self.batchnorm = BatchNormalization(trainable=False)
        self.maxpool = MaxPool2D((3, 3), strides=(2, 2), padding="same")
        self.zero_pad = ZeroPadding2D((3,3))
        self.activation = Activation('relu')

    @staticmethod
    def conv_block(input_tensor, kernel_size, filters, use_bias=True, training=None):
        filter1, filter2, filter3 = filters
        x = Conv2D(filter1, (1, 1), strides=(2, 2), use_bias=use_bias)(input_tensor)
        x = BatchNormalization()(x, training=training)
        x = Activation('relu')(x)
        x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', use_bias=use_bias)(x)
        x = BatchNormalization()(x, training=training)
        x = Activation('relu')(x)
        x = Conv2D(filter3, (1, 1), use_bias=use_bias)(x)
        x = BatchNormalization()(x, training=training)
        shortcut = Conv2D(filter3, (1, 1), strides=(2, 2), use_bias=True)(input_tensor)
        shortcut = BatchNormalization()(shortcut, training=training)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    @staticmethod
    def identity_block(input_tensor, kernel_size, filters, use_bias=True, training=None):
        filter1, filter2, filter3 = filters
        x = Conv2D(filter1, (1, 1), use_bias=use_bias)(input_tensor)
        x = BatchNormalization()(x, training=training)
        x = Activation('relu')(x)
        x = Conv2D(filter2, (kernel_size, kernel_size), padding='same', use_bias=True)(x)
        x = BatchNormalization()(x, training=training)
        x = Activation('relu')(x)
        x = Conv2D(filter3, (1, 1), use_bias=True)(x)
        x = BatchNormalization()(x, training=training)
        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def resnet50(self, input_image, training=None):
        x = self.zero_pad(input_image)
        x = self.conv_1(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        C1 = x = self.maxpool(x)
        x = self.conv_block(x, 3, [64, 64, 256], training=training)
        x = self.identity_block(x, 3, [64, 64, 256], training=training)
        C2 = x = self.identity_block(x, 3, [64, 64, 256], training=training)
        x = self.conv_block(x, 3, [128, 128, 512], training=training)
        x = self.identity_block(x, 3, [128, 128, 512], training=training)
        x = self.identity_block(x, 3, [128, 128, 512], training=training)
        C3 = x = self.identity_block(x, 3, [128, 128, 512], training=training)
        x = self.conv_block(x, 3, [256, 256, 1024], training=training)
        # If we wanted to change this to resnet 101, we'd have 22 instead of 5 blocks between here and C4
        x = self.identity_block(x, 3, [256, 256, 1024], training=training)
        x = self.identity_block(x, 3, [256, 256, 1024], training=training)
        x = self.identity_block(x, 3, [256, 256, 1024], training=training)
        x = self.identity_block(x, 3, [256, 256, 1024], training=training)
        x = self.identity_block(x, 3, [256, 256, 1024], training=training)
        C4 = x
        x = self.conv_block(x, 3, [512, 512, 2048], training=training)
        x = self.identity_block(x, 3, [512, 512, 2048], training=training)
        C5 = self.identity_block(x, 3, [512, 512, 2048], training=training)
        return [C1, C2, C3, C4, C5]


class MaskRCNN(Model):
    """Mask R-CNN implementation built on top of ResNet50"""
    def __init__(self, mode, config, model_dir):
        super(MaskRCNN, self).__init__()
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self._anchor_cache = None
        self.anchors = None
        self.epoch = 0
        self.checkpoint_path = None
        self.log_dir = None
        self.set_log_dir()
        self.model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """ Build is called in __init__ for the MaskRCNN model. It does some checking on the images to ensure that they
        have an appropriate dimension.
        Given appropriate images, it will generate the Mask R-CNN architecture."""
        h, w = config.IMAGE_SHAPE[:2]
        if h/2**6 != int(h/2**6) or w/2**6 != int(w/2**6):
            raise Exception("Image size must be a multiple of 64 to allow up and downscaling")

        input_image = Input(shape=[None, None, config.IMAGE_SHAPE[2]])
        input_image_meta = Input(shape=[config.IMAGE_META_SIZE])

        # First, we want to consider whether we're in training mode or inference mode. IF we're in training mode,
        # we need to build Input layers that we'll feed to the model, which includes all of the class IDs, Bounding
        # Boxes for the region proposal network.
        if mode == "training":
            input_rpn_match = Input(shape=[None, 1], dtype=tf.int32)
            input_rpn_bbox = Input(shape=[None, 4], dtype=tf.float32)

            # We're not *super* concerned with class id in the WFP use case, but it's included.
            input_gt_class_ids = Input(shape=[None], dtype=tf.int32)
            input_gt_boxes = Input(shape=[None, 4], dtype=tf.float32)

            gt_boxes = Lambda(lambda x: mp_utils.norm_boxes(x, tf.shape(input_image)[1:3]))(input_gt_boxes)
            input_gt_masks = Input(shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None], dtype=bool)

        elif mode == "inference":
            input_anchors = Input(shape=[None, 4])

        # Once we have our inputs set up, we need to build our backbone model. This will convert it from a standard
        # image (RGB or NVDI) - 13 channels is preferable, but it's an easy modification.
        backbone_model = Backbone()
        _, C2, C3, C4, C5 = backbone_model.resnet50(input_image, training=config.TRAINABLE)

        # We then use the output of the ResNet model to construct the Feature Pyramid Network.
        # The feature pyramid network is used to represent single objects at multiple scales.
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
        else:
            anchors = input_anchors

        # Using the anchors pulled from above, we build our region proposal network.
        # The region proposal network (RPN) scans the image like sliding a small window across the whole image to find
        # areas that contain the objects in question. We call these areas "anchors" and there are thousands of them
        # which overlap to cover the entire image.
        rpn = self.build_rpn_model(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
        layer_outputs = []
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bounding_box"]
        outputs = list(zip(*layer_outputs))
        outputs = [tf.keras.layers.Concatenate(axis=1, name=n)(list(o)) for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bounding_box = outputs

        # The ProposalLayer here is a custom Keras layer (written by MatterPort) which reads the output of our
        # region proposal network, picks the top anchors, and adjusts the size of the bounding box based on whether
        # the anchor identified is in the foreground or the background of the image and how much the RPN
        # estimates the size of the object to be off by. This identifies our regions of interest (ROIs).
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(proposal_count=proposal_count, nms_threshold=config.RPN_NMS_THRESHOLD, config=config)\
            ([rpn_class, rpn_bounding_box, anchors])

        if mode == "training":

            active_class_ids = Lambda(lambda x: mp_utils.parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)
            target_rois = rpn_rois

            # noinspection PyUnboundLocalVariable
            rois, target_class_ids, target_bounding_box, target_mask = DetectionTargetLayer(config)\
                ([target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # The feature proposal network classifier looks at the regions of interest proposed by the region proposal
            # network and generates a class for the object (which we only have 2 at the time, but if we add more crops
            # in the future, these are the classes which will need to be changed.) and a bounding box refinement to
            # narrow down the size of the bounding box.
            # This is also where region of interest pooling occurs. This is critical to do because classifiers are not
            # intended to deal with variable input size. Since ROI boxes can have different sizes coming out of the RPN,
            # we "crop" the feature map and set it to a fixed size.
            # We sample the feature map at different points and apply a bilinear interpolation
            mask_rcnn_class_logits, mask_rcnn_class, mask_rcnn_bounding_box = \
                self.fpn_classifier(rois, mask_rcnn_feature_maps, input_image_meta, config.POOL_SIZE,
                                    config.NUM_CLASSES, training=config.TRAINABLE,
                                    layer_size=config.FPN_FC_LAYERS_SIZE)

            mask_rcnn_mask = self.build_fpn_mask(rois, mask_rcnn_feature_maps, input_image_meta,
                                                 config.MASK_POOL_SIZE, config.NUM_CLASSES, training=config.TRAINABLE)

            output_rois = tf.identity(rois)

            # Once we have our ROIs, masks, and and so on, we need to identify our losses, which are Lambda layers
            # noinspection PyUnboundLocalVariable
            rpn_class_loss = Lambda(lambda x: mp_utils.calculate_rpn_class_loss(*x))([input_rpn_match, rpn_class_logits])
            rpn_bounding_loss = Lambda(lambda x: mp_utils.calculate_rpn_bounding_loss(config, *x))\
                ([input_rpn_bbox, input_rpn_match, rpn_bounding_box])
            class_loss = Lambda(lambda x: mp_utils.mask_rcnn_class_loss(*x))\
                ([target_class_ids, mask_rcnn_class_logits, active_class_ids])
            bounding_loss = Lambda(lambda x: mp_utils.mask_rcnn_bounding_loss(*x))\
                ([target_bounding_box, target_class_ids, mask_rcnn_bounding_box])
            mask_loss = Lambda(lambda x: mp_utils.mask_rcnn_mask_loss(*x))([target_mask, target_class_ids, mask_rcnn_mask])

            # Finally, we use the inputs and outputs we've built to construct a model.
            inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox, input_gt_class_ids,
                      input_gt_boxes, input_gt_masks]
            outputs = [rpn_class_logits, rpn_class, rpn_bounding_box, mask_rcnn_class_logits, mask_rcnn_class,
                       mask_rcnn_bounding_box, mask_rcnn_mask, rpn_rois, output_rois, rpn_class_loss, rpn_bounding_loss,
                       class_loss, bounding_loss, mask_loss]

            model = Model(inputs, outputs)

        else:
            # In inference mode, we still build the fpn classifier and return a model with all of the inputs and outputs
            # but we don't need all of the extra stuff (like losses)
            mask_rcnn_class_logits, mask_rcnn_class, mask_rcnn_bounding_box = \
                self.fpn_classifier(rpn_rois, mask_rcnn_feature_maps, input_image_meta, config.POOL_SIZE,
                                    config.NUM_CLASSES, training=config.TRAINABLE,
                                    layer_size=config.FPN_FC_LAYERS_SIZE)
            detections = DetectionLayer(config)([rpn_rois, mask_rcnn_class, mask_rcnn_bounding_box, input_image_meta])
            detection_boxes = Lambda(lambda x: x[..., :4])(detections)
            mask_rcnn_mask = self.build_fpn_mask(detection_boxes, mask_rcnn_feature_maps, input_image_meta,
                                                 config.MASK_POOL_SIZE, config.NUM_CLASSES, training=config.TRAINABLE)
            model = Model([input_image, input_image_meta, input_anchors], [detections, mask_rcnn_class,
                                                                           mask_rcnn_bounding_box, mask_rcnn_mask,
                                                                           rpn_rois, rpn_class, rpn_bounding_box])

        return model

    # TODO: Define losses internally since we have so many.

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

    def set_log_dir(self, model_path=None):
        self.epoch = 0
        now = datetime.datetime.now()

        if model_path:
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

            # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def load_weights(self, file_path, by_name=False, exclude=None):
        import h5py
        from tensorflow.keras import saving

        if exclude:
            by_name = True

        f = h5py.File(file_path, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        model = self.model
        layers = model.inner_model.layers if hasattr(model, "inner_model") else model.layers

        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)
        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_attributes_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        self.set_log_dir(file_path)

    @staticmethod
    def compute_backbone_shapes(config, image_shape):
        return np.array([[int(np.ceil(image_shape[0] / stride)), int(np.ceil(image_shape[1] / stride))]
                         for stride in config.BACKBONE_STRIDES])

    def get_anchors(self, image_shape):
        backbone_shapes = self.compute_backbone_shapes(self.config, image_shape)
        if self._anchor_cache is None:
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            a = mp_utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            self.anchors = a
            self._anchor_cache[tuple(image_shape)] = mp_utils.norm_boxes(a, image_shape[:2])
        return self._anchor_cache[tuple(image_shape)]

    @staticmethod
    def fpn_classifier(rois, feature_maps, image_meta, pool_size, num_classes, training=None, layer_size=1024):
        x = PyramidROIAlign([pool_size, pool_size])([rois, image_meta] + feature_maps)
        x = TimeDistributed(Conv2D(layer_size, (pool_size, pool_size), padding="VALID"))(x)
        x = TimeDistributed(BatchNormalization())(x, training=training)
        x = Activation('relu')(x)
        x = TimeDistributed(Conv2D(layer_size, (1, 1)))(x)
        x = TimeDistributed(BatchNormalization())(x, training=training)
        x = Activation('relu')(x)
        shared = Lambda(lambda x: tf.squeeze(tf.squeeze(x, 3), 2))(x)

        mask_rcnn_class_logits = TimeDistributed(Dense(num_classes))(shared)
        mask_rcnn_probabilities = TimeDistributed(Activation("softmax"))(mask_rcnn_class_logits)

        x = TimeDistributed(Dense(num_classes * 4, activation='linear'))(shared)
        s = tf.keras.backend.int_shape(x)
        mask_rcnn_bounding_box = Reshape((s[1], num_classes, 4))(x)

        return mask_rcnn_class_logits, mask_rcnn_probabilities, mask_rcnn_bounding_box

    @staticmethod
    def build_fpn_mask(rois, feature_maps, image_meta, pool_size, num_classes, training=None):
        # Start with ROI pooling
        x = PyramidROIAlign([pool_size, pool_size])([rois, image_meta] + feature_maps)
        # 4 normal conv layers, a deconv, and a conv with sigmoid.
        x = TimeDistributed(Conv2D(256, (3, 3), padding="SAME"))(x)
        x = TimeDistributed(BatchNormalization())(x, training=training)
        x = Activation('relu')(x)
        x = TimeDistributed(Conv2D(256, (3, 3), padding="SAME"))(x)
        x = TimeDistributed(BatchNormalization())(x, training=training)
        x = Activation('relu')(x)
        x = TimeDistributed(Conv2D(256, (3, 3), padding="SAME"))(x)
        x = TimeDistributed(BatchNormalization())(x, training=training)
        x = Activation('relu')(x)
        x = TimeDistributed(Conv2D(256, (3, 3), padding="SAME"))(x)
        x = TimeDistributed(BatchNormalization())(x, training=training)
        x = Activation('relu')(x)
        x = TimeDistributed(Conv2DTranspose(256, (2, 2), strides=2, activation="relu"))(x)
        x = TimeDistributed(Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"))(x)
        return x


    @staticmethod
    def build_rpn_model(anchor_stride, anchors_per_location, depth):
        input_feature_map = Input(shape=[None, None, depth])

        shared = Conv2D(512, (3, 3), padding='same', activation='relu', strides=anchor_stride)(input_feature_map)
        x = Conv2D(2 * anchors_per_location, (1, 1), padding='VALID', activation='linear')(shared)
        rpn_class_logits = Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)
        rpn_probabilities = Activation("softmax")(rpn_class_logits)

        x = Conv2D(anchors_per_location * 4, (1, 1), padding="VALID", activation='linear')(shared)
        rpn_bounding_box = Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

        outputs = [rpn_class_logits, rpn_probabilities, rpn_bounding_box]

        return Model([input_feature_map], outputs)

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.
        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = mp_utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mp_utils.mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = mp_utils.compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.
        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.
        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks

        This function copied directly from Matterport implementation.
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = mp_utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = mp_utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = mp_utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def train(self, x_train, y_train, validation_data, learning_rate, epochs):
        """ Train the Mask R-CNN model on the provided training and validation data sets using the model.fit method"""
        assert self.mode == "training", "Model must be in training mode"

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        print(f"Starting to train! Learning rate is {learning_rate}")
        print(f"Path to checkpoints is: {self.checkpoint_path}")

        callbacks = [tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, save_weights_only=True),
                     tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True,
                                                           write_images=False)]

        self.model.fit(x=x_train, y=y_train, epochs=epochs, callbacks=callbacks, validation_data=validation_data)
        self.epoch = max(self.epoch, epochs)

    def detect(self, images):
        assert self.mode == "inference", "Model must be in inference mode"
        assert len(images) == self.config.BATCH_SIZE, "Number of images must be equal to BATCH_SIZE"

        molded_images, image_metas, windows = self.mold_inputs(images)

        image_shape = images[0].shape

        anchors = self.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        detections, _, _, mask_rcnn_mask, _, _, _ = self.model.predict([images, image_metas, anchors])

        results = []

        for i, image in enumerate(images):
            rois, class_ids, scores, masks = self.unmold_detections(detections[i], mask_rcnn_mask[i], image.shape,
                                                                    windows[i])
            results.append({"rois": rois, "class_ids": class_ids, "scores": scores, "masks": masks})

        return results