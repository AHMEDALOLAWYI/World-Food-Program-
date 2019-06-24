import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras





def down_layer(input_layers, filters,pool=True):
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
    residual = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    if pool:
        max_pool = MaxPool2D()(residual)
        return max_pool, residual
    else:
        return residual

#upsampling convolution
def up(input_layer, residual, filters):
    filters=int(filters)
    upsample = UpSampling2D()(input_layer)
    upconv = Conv2D(filters, kernel_size=(2, 2), padding="same",activation='relu')(upsample)
    concat = Concatenate(axis=3)([residual, upconv])
    conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat)
    conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
    return conv2

def unet(filters=64,input_size=(256,256,1)):
    inputs = Input(input_size)
    layers=[input]
    residuals = []

    #first downsampling
    d1, res1 = down(input_layer, filters)
    residuals.append(res1)
    #filters are multiplied by two after each downsampling
    filters *= 2

    #second downsampling
    d2, res2 = down(d1, filters)
    residuals.append(res2)
    filters *= 2

    #third downsampling
    d3, res3 = down(d2, filters)
    residuals.append(res3)
    filters *= 2

    #fourth downsampling
    d4, res4 = down(d3, filters)
    residuals.append(res4)
    filters *= 2

    #fifth downsampling without max pool
    d5 = down(d4, filters, pool=False)
    #first upsampling and combination with latest residual
    up1 = up(d5, residual=residuals[-1], filters=filters/2)
    filters /= 2

    up2 = up(up1, residual=residuals[-2], filters=filters / 2)
    filters /= 2

    # Up 3, 64
    up3 = up(up2, residual=residuals[-3], filters=filters / 2)
    filters /= 2

    # Up 4, 128
    up4 = up(up3, residual=residuals[-4], filters=filters / 2)
    out = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(up4)

    model = Model(input_layer, out)
    return model


def dice_coef(y_true, y_pred):
    smooth = 1e-5

    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))

    isct = tf.reduce_sum(y_true * y_pred)

    return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

model = get_unet_model(filters=64)
model.summary()
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[dice_coef])
model.fit_generator(train_gen, steps_per_epoch=10, epochs=10)

