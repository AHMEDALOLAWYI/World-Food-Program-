from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Flatten,
    merge,
    Lambda
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D,
    ZeroPadding2D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

def BNConv(nb_filter, nb_row, nb_col, w_decay, subsample=(1, 1), border_mode="same"):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                      border_mode=border_mode, activation="relu",
                      W_regularizer=l2(w_decay) if w_decay else None, init="he_normal")(input)
        return BatchNormalization(mode=0, axis=1)(conv)
    return f


def inception_v3(w_decay=None):
    input = Input(shape=(3, 299, 299))

    conv_1 = BNConv(32, 3, 3, w_decay, subsample=(2, 2), border_mode="valid")(input)
    conv_2 = BNConv(32, 3, 3, w_decay, border_mode="valid")(conv_1)
    conv_3 = BNConv(64, 3, 3, w_decay)(conv_2)
    pool_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="valid")(conv_3)

    conv_5 = BNConv(80, 1, 1, w_decay)(pool_4)
    conv_6 = BNConv(192, 3, 3, w_decay, border_mode="valid")(conv_5)
    pool_7 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="valid")(conv_6)

    inception_8 = InceptionFig5(w_decay)(pool_7)
    inception_9 = InceptionFig5(w_decay)(inception_8)
    inception_10 = InceptionFig5(w_decay)(inception_9)

    inception_11 = DimReductionA(w_decay)(inception_10)

    inception_12 = InceptionFig6(w_decay)(inception_11)
    inception_13 = InceptionFig6(w_decay)(inception_12)
    inception_14 = InceptionFig6(w_decay)(inception_13)
    inception_15 = InceptionFig6(w_decay)(inception_14)
    inception_16 = InceptionFig6(w_decay)(inception_15)

    inception_17 = DimReductionB(w_decay)(inception_16)

    inception_18 = InceptionFig7(w_decay)(inception_17)
    inception_19 = InceptionFig7(w_decay)(inception_18)

    pool_20 = Lambda(lambda x: K.mean(x, axis=(2, 3)), output_shape=(2048, ))(inception_19)

    model = Model(input, pool_20)

    return model
