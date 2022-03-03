""" This module contains setups to configure tensorflow. """
import tensorflow as tf

TF_DTYPE_INT = tf.int32
TF_DTYPE = tf.float32

EPS = 1e-6


def float_me(x):
    return tf.cast(x, dtype=TF_DTYPE)


EPS_TF = float_me(EPS)


def int_me(x):
    return tf.cast(x, dtype=TF_DTYPE_INT)


def set_manual_seed_tf(seed):
    """Set libraries random seed for reproducibility."""
    tf.random.set_seed(seed)
