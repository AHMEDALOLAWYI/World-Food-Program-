#!/usr/bin/env python
# Variational Autoencoder model for Omdena SuperResolution task
# Last updated 29 June 2019
# Credit for the CVAE code itself to the TensorFlow team - https://www.tensorflow.org/beta/tutorials/generative/cvae

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Set program-wide variables
in_shape = (130, 130, 3)
num_units = (130 * 7 * 7)

image_path = None
hi_res_image_path = None

epochs = 1000
latent_dim = 50
examples_to_generate = 10

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=in_shape)
                layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                layers.conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                layers.Flatten()
                layers.Dense(latent_dim * 2)
            ]
        )
        self.generative_net = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(latent_dim,)),
                layers.Dense(units=num_units, activation=tf.nn.relu),
                layers.Reshape(target_shape=num_units),
                layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
                layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME")
            ]
        )

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparametrize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits


def log_normal_pdf(sample, mean, logvar):
    log2pi = tf.math.log(2 * np.pi)
    return tf.reduce_sum(-0.5 * (((sample - mean) ** 2) * tf.exp(-logvar) + logvar + log2pi), axis=1)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparametrize(mean, logvar)
    x_logits = model.decode(z)

    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=x)
    logpx_z = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0, 0)
    logqz_x = log_normal_pdf(z, mean, logvar)

    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    return tape.gradient(loss, model.trainable_variables), loss


def apply_gradients(opt, grads, vars):
    opt.apply_gradients(zip(grads, vars))


if __name__ == "__main__":
    vector_for_generation = tf.random.normal(shape=[examples_to_generate, latent_dim])
    model = CVAE(latent_dim)
    # TODO: Finish writing the model to generate images
    # TODO: Add commentary so that this is useful for teaching.