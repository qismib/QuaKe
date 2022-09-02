""" This module contains setups to configure tensorflow. """
from typing import Union
import tensorflow as tf
import numpy as np

TF_DTYPE_INT = tf.int32
TF_DTYPE = tf.float32
TF_DTYPE_BOOL = tf.bool

EPS = 1e-6

arrayLike = Union[tf.Tensor, np.ndarray]


def float_me(x: arrayLike) -> tf.Tensor:
    """Casts array to tf.Tensor and dtype `TF_DTYPE`.

    Parameters
    ----------
    x: arrayLike
        The input array or sequence.

    Returns
    -------
    tf.Tensor
        The tensor sequence.
    """
    return tf.cast(x, dtype=TF_DTYPE)


EPS_TF = float_me(EPS)

TF_PI = float_me(np.pi)


def int_me(x: arrayLike) -> tf.Tensor:
    """Casts array to tf.Tensor and dtype `TF_DTYPE_INT`.

    Parameters
    ----------
    x: arrayLike
        The input array or sequence.

    Returns
    -------
    tf.Tensor
        The tensor sequence.
    """
    return tf.cast(x, dtype=TF_DTYPE_INT)


def bool_me(x: arrayLike) -> tf.Tensor:
    """Casts array to tf.Tensor and dtype `TF_DTYPE_BOOl`.

    Parameters
    ----------
    x: arrayLike
        The input array or sequence.

    Returns
    -------
    tf.Tensor
        The tensor sequence.
    """
    return tf.cast(x, dtype=TF_DTYPE_BOOL)


def set_manual_seed_tf(seed: int):
    """Set tensorflow library random seed for reproducibility.

    Parameters
    ----------
    seed: int
        Random generator seed for code reproducibility.
    """
    tf.random.set_seed(seed)
