""" This module contains utility functions common to all models."""
import logging
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from quake import PACKAGE

logger = logging.getLogger(PACKAGE)


def dataset_split_util(
    data: np.ndarray,
    labels: np.ndarray,
    split_ratio: float,
    seed: int = 42,
    with_indices: bool = False,
) -> list[list[np.ndarray]]:
    """Split dataset and labels into train test and validation sets.

    Test and split ratio have the same factor overall.

    Parameters
    ----------
    data: np.ndarray
        Batched data, of shape=(nb events, ...)
    labels: nb.ndarray
        Array of labels, of shape=(nb events,)
    split_ratio: float
        The test and validation splitting percentage
    seed: int
        Random generator seed for reproducibility.
    with_indices: bool
        Wether to return the indices of the selected train, test, validation
        examples for reporoducibility.

    Returns
    -------
    List[np.ndarray]
        Training set (data, labels, [indices])
    List[np.ndarray]
        Validation set (data, labels, [indices])
    List[np.ndarray]
        Test set (data, labels, [indices])
    """
    arrays = [data, labels]
    if with_indices:
        idxs = np.arange(data.shape[0])
        arrays.append(idxs)

    first_split = train_test_split(
        *arrays, test_size=2 * split_ratio, random_state=seed
    )
    train_arrays = first_split[::2]
    test_val_arrays = first_split[1::2]

    second_split = train_test_split(*test_val_arrays, test_size=0.5, random_state=seed)
    val_arrays = second_split[::2]
    test_arrays = second_split[1::2]

    return train_arrays, val_arrays, test_arrays


def get_dataset_balance_message(dataset: tf.keras.utils.Sequence, name: str):
    """
    Logs the dataset balancing between classes
    Parameters
    ----------
    dataset: tf.keras.utils.Sequence
        The dataset to log.
    name: str
        The dataset name to be logged.

    Returns
    -------
    msg: str
        The balance message to be printed.
    """
    nb_examples = dataset.data_len
    positives = np.count_nonzero(dataset.targets)
    msg = (
        f"{name} dataset balancing: {nb_examples} training points, "
        f"of which {positives/nb_examples*100:.2f}% positives"
    )
    return msg
