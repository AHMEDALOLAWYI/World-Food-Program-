#!/usr/bin/env python
# Omdena Mask Regional Convolutional Neural Network (Mask R-CNN) for UN World Food Program
# Written by Erick Galinkin
# Note: I've opted to use Tensorflow 2.0 here. It should be easy to change if need be, but I'm assuming
# that support for TF2 will be longer-lived, and we get to play with all of the Keras abstractions without having to
# import a second library.

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Flatten, Conv2D, BatchNormalization, Add, Activation, ZeroPadding2D, MaxPool2D
from tensorflow.python.keras import Model

data_dir = './images/'


# We need to add a lot more to make this work - right now it's just a conventional CNN.
class MaskRCNN(Model):
    def __init__(self):
        super(MaskRCNN, self).__init__()
        self.conv1 = Conv2D(256, (1, 1), use_bias=True)
        self.conv2 = Conv2D(128, (3, 3), padding='same', use_bias=True)
        self.conv3 = Conv2D(64, (1, 1), use_bias=True)
        self.bn_train = BatchNormalization(trainable=True)
        self.bn_freeze = BatchNormalization(trainable=False)
        self.flatten = Flatten()
        self.add = Add()
        self.activation = Activation('relu')


# It may be prudent to rip this whole thing out and use Thomas's pre-trained Resnet-50
# Heavy inspiration from Matterport's Mask RCNN implementation.
class Backbone(Model):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv_1 = Conv2D(64, (7, 7), strides=(2, 2), use_bias=True)
        self.batchnorm = BatchNormalization(trainable=True)
        self.maxpool = MaxPool2D((3, 3), strides=(2, 2), padding="same")
        self.zero_pad = ZeroPadding2D((3,3))
        self.activation = Activation('relu')

    @staticmethod
    def conv_block(input_tensor, kernel_size, filters, use_bias=True, trainable=True):
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
    def identity_block(input_tensor, kernel_size, filters, use_bias=True, trainable=True):
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

    def resnet_50(self, channels, input_image, trainable):
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
        # Skipping stage 5 entirely - if we decide we need it, we can always come back and add it.
        return [C1, C2, C3, C4]



# There is no labeled data yet, so this is more or less just a mock-up. Once we have data, we'll read it in from a
# path and then return whatever format we need.
def ingest_data(path):
    pass
    return None


# Once again, once we know what format our data is in, we can fill this bit out. It will probably be some kind of
# wrapper around sklearn's train_test_split, but that's more or less an implementation detail.
def split_data(data):
    pass
    return None


def train_model(images, labels, epochs):
    model = MaskRCNN()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_loss = tf.keras.metrics.Mean(name='training_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    optimizer = tf.keras.optimizers.SGD()
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
    EPOCHS = 10

    path_to_data = None  # TODO: Find the best way to populate this path
    data = ingest_data(path_to_data)
    X_train, X_val, y_train, y_val = split_data(data)
    model = train_model(X_train, y_train, EPOCHS)
    validation_loss, validation_accuracy = validate_model(model, X_val, y_val)
    print('Validation loss: {}, Validation accuracy: {}'.format(
        validation_loss.result(), validation_accuracy.result()*100))
