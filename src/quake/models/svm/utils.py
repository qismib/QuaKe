"""This module implements the utility functions for SVM training."""
from typing import Tuple
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from ..AbstractNet import AbstractNet, FeatureReturner


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
    with FeatureReturner(network) as fr:
        _, features = fr.predict(generator, verbose=1)
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


def rearrange_scale(
    train_set: np.ndarray,
    val_set: np.ndarray,
    test_set: np.ndarray,
    should_do_scaling: bool,
) -> list[np.ndarray]:
    """Utility function to provide scaling.

    Rearranges the input data in a unique list and applies a scaling for enhancing SVMs performances.

    Parameters
    ----------
    train_set: np.ndarray
        The training array, of shape=(nb events, nb features).
    val_set: np.ndarray
        The validation array, of shape=(nb events, nb features).
    test_set: np.ndarray
        The test array, of shape=(nb events, nb features).
    should_do_scaling: bool
        Wether to do input scaling or not.

    Returns
    -------
    dataset: list[np.ndarray]
        The scaled inputs if `should_do_scaling` is True, the inputs themselves
        otherwise.
    """
    full_dataset = [train_set, val_set, test_set]
    if should_do_scaling:
        #quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
        #quantile_transformer = preprocessing.PowerTransformer(standardize = True)
        #scaler = quantile_transformer.fit(np.concatenate(full_dataset))
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(
            np.concatenate(full_dataset)
        )
        for i in range(3):
            full_dataset[i] = scaler.transform(full_dataset[i])
    return full_dataset
