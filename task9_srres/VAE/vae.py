#!/usr/bin/env python
# Variational Autoencoder model for Omdena SuperResolution task
# Last updated 29 June 2019
# Credit for the CVAE code itself to the TensorFlow team - https://www.tensorflow.org/beta/tutorials/generative/cvae

import tensorflow as tf
from tensorflow.python.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Set program-wide variables
in_shape = (64, 64, 3)
num_units = (64 * 7 * 7)

path_to_images = "../megatools/highres-2000/"

epochs = 1000
latent_dim = 50
examples_to_generate = 10


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=in_shape),
                layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
                layers.Flatten(),
                layers.Dense(latent_dim * 2)
            ]
        )
        self.generative_net = tf.keras.Sequential(
            [
                layers.InputLayer(input_shape=(latent_dim,)),
                layers.Dense(units=num_units, activation=tf.nn.relu),
                layers.Reshape(target_shape=(7, 7, 64)),
                layers.Conv2DTranspose(filters=128, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
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


def generate_images(model, epoch, test_input, fname):
  predictions = model.sample(test_input)
  plt.figure(figsize=(64,64))

  for i in range(predictions.shape[0]):
      plt.subplot(64, 64, i+1)
      plt.imshow(predictions[i, :, :, 0])
      plt.axis('off')

  plt.savefig('./vae_output/{}_{}.png'.format(fname, epoch))


if __name__ == "__main__":
    vector_for_generation = tf.random.normal(shape=[examples_to_generate, latent_dim])
    model = CVAE(latent_dim)
    optimizer = tf.keras.optimizers.Adam()
    generate_images(model, 0, vector_for_generation, "test")
    # TODO: Do this for lots of images instead of just one
    train_x = path_to_images + "00075.png"
    test_x = path_to_images + "00151.png"
    for epoch in range(1, epochs+1):
        for x in train_x:
            gradients, loss = compute_gradients(model, x)
            apply_gradients(optimizer, gradients, model.trainable_variables)

        loss = tf.keras.metrics.Mean()
        for x in test_x:
            loss(compute_loss(model, x))
        elbo = -loss.result()
        print("Epoch: {}, Test set ELBO: {}".format(epoch, elbo))

        generate_images(model, epoch, vector_for_generation, "test")
    # TODO: Add commentary so that this is useful for teaching.
