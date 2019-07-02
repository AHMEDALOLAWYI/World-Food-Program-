import warnings
from functools import wraps

import numpy as np
from keras.layers import BatchNormalization
from keras.models import model_from_json


def legacy_support(kwargs_map):
    """
    Decorator which map old kwargs to new ones
    Args:
        kwargs_map: dict 'old_argument: 'new_argument' (None if removed)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            # rename arguments
            for old_arg, new_arg in kwargs_map.items():
                if old_arg in kwargs.keys():
                    if new_arg is None:
                        raise TypeError(
                            "got an unexpected keyword argument '{}'".format(old_arg)
                        )
                    warnings.warn(
                        "`{old_arg}` is deprecated and will be removed "
                        "in future releases, use `{new_arg}` instead.".format(
                            old_arg=old_arg, new_arg=new_arg
                        )
                    )
                    kwargs[new_arg] = kwargs[old_arg]

            return func(*args, **kwargs)

        return wrapper

    return decorator


def freeze_model(model):
    """model all layers non trainable, excluding BatchNormalization layers"""
    for layer in model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    return


def get_layer_number(model, layer_name):
    """
    Help find layer in Keras model by name
    Args:
        model: Keras `Model`
        layer_name: str, name of layer
    Returns:
        index of layer
    Raises:
        ValueError: if model does not contains layer with such name
    """
    for i, l in enumerate(model.layers):
        if l.name == layer_name:
            return i
    raise ValueError(
        "No layer with name {} in  model {}.".format(layer_name, model.name)
    )


def to_tuple(x):
    if isinstance(x, tuple):
        if len(x) == 2:
            return x
    elif np.isscalar(x):
        return (x, x)

    raise ValueError(
        'Value should be tuple of length 2 or int value, got "{}"'.format(x)
    )
