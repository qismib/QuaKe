"""This module implements the utility functions for SVM training."""
from typing import Tuple
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from ..attention.AbstractNet import AbstractNet


def extract_feats(
    generator: tf.keras.utils.Sequence,
    network: AbstractNet,
    should_add_extra_feats: bool,
    should_remove_outliers: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts features from each event in the given dataset.

    Parameters
    ----------
    generator: tf.keras.utils.Sequence
        The dataset generator.
    network: AbstractNet
        The feature extractor network.
    should_add_extra_feats: bool
        Wether to enhance extracted features with custom ones.
    should_remove_outliers: bool
        Wether to remove outlier events or not.

    Returns
    -------
    features: np.ndarray
        The extracted features, of shape=(nb events, nb features).
    labels: np.ndarray
        The labels array, of shape=(nb events,).
    """
    features = network.predict_and_extract(generator)[1].numpy()
    labels = generator.targets

    # optional adding custom extra features
    if should_add_extra_feats:
        extra_features = generator.get_extra_features()
        features = np.concatenate([features, extra_features], axis=-1)

    # remove outlier training examples
    if should_remove_outliers:
        features, labels = remove_outliers(features, labels)
    return features, labels


def remove_outliers(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Removes outlier events from a dataset.

    An outlier is defined as an event, whose features lie further than 3.5
    standard deviations from the mean. This procedure is useful to clean
    examples from the SVM training set.

    Parameters
    ----------
    features: np.ndarray
        The array of features.
    labels: np.ndarray
        The array of labels.

    Returns
    -------
    bulk_events: np.ndarray
        The outlier filtered events.
    """
    mus = features.mean(axis=0, keepdims=True)
    sigmas = sigmas.mean(axis=0, keepdims=True)
    good_examples = (features - mus) / sigmas < 3.5
    # TODO: maybe change the rule for selecting an outlier
    # compute euclidean distance from all the feature means and put it < 3.5
    good_examples = np.all(good_examples, axis=1)
    return features[good_examples], labels[good_examples]


def scaler(inputs: np.ndarray, kernel: str, should_do_scaling: bool):
    """Utility function to provide scaling depending on selected kernel.

    Transforms the input data for enhancing SVMs performances.

    Parameters
    ----------
    inputs: np.ndarray
        The input array, of shape=(nb events, nb features).
    kernel: str
        The kernel label.
    should_do_scaling: bool
        Wether to do input scaling or not.

    Returns
    -------
    dataset: np.ndarray
        The scaled inputs if `should_do_scaling` is True, the inputs themselves
        otherwise.
    """
    if should_do_scaling:
        if kernel == "linear" or kernel == "rbf":
            scaler = preprocessing.PowerTransformer(
                standardize=True
            )  # nonlinear map to gaussian
        elif kernel == "poly":
            scaler = preprocessing.QuantileTransformer(
                random_state=0
            )  # nonlinear map to uniform dist
        else:
            NotImplementedError(f"scaler not implemented for {kernel} kernel")
        return scaler.fit_transform(inputs)
    return inputs
