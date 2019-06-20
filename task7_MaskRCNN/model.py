#!/usr/bin/env python
# Omdena Mask R-CNN for UN World Food Program
# Written by Erick Galinkin
# Note: I've opted to use Tensorflow 2.0 here. It should be easy to change if need be, but I'm assuming
# that support for TF2 will be longer-lived, and we get to play with all of the Keras abstractions without having to
# import a second library.

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


# We need to add a lot more to make this work - right now it's just a conventional CNN.
class MaskRCNN(Model):
    def __init__(self):
        super(MaskRCNN, self).__init__()
        self.conv = Conv2D(256, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


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
