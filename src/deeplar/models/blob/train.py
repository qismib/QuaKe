""" This module provides functions for Blob method classiication."""
import logging
from typing import Tuple
from time import time as tm
from pathlib import Path
from deeplar.models.attention.attention_dataloading import read_data, Dataset
from deeplar import PACKAGE
from deeplar.utils.diagnostics import (
    save_histogram_activations_image,
    save_scatterplot_features_image,
)
import numpy as np
from deeplar.models.blob.blob_detection import get_blob_energies, train_blobs

logger = logging.getLogger(PACKAGE + ".blob")


def make_inference_plots(train_folder: Path, features: np.ndarray, labels: np.ndarray):
    """Plotting blob energies and saving the figure.

    Parameters:
    -----------
    train_folder: Path
        The train output folder path.
    features: np.ndarray
        Blob energies.
    labels: np.ndarray
        Truth labels.
    """
    fname = train_folder / "scatterplot_features.svg"
    save_scatterplot_features_image(fname, features, labels)


def blob_train(data_folder: Path, train_folder: Path, setup: dict):
    """Computing blobs and classifying using Neyman-Pearson's lemma or with an SVM.

    Parameters
    ----------
    data_folder: Path
        The input data folder path.
    train_folder: Path
        The train output folder path.
    setup: dict
        Settings dictionary.
    """

    # data loading
    train_generator, val_generator, test_generator = read_data(
        data_folder, train_folder, setup
    )

    logger.info("Detecting blobs and measuring energies ...")
    train_features, train_labels = (
        np.array(get_blob_energies(train_generator.inputs, setup)).T,
        train_generator.targets,
    )
    val_features, val_labels = (
        np.array(get_blob_energies(val_generator.inputs, setup)).T,
        val_generator.targets,
    )
    test_features, test_labels = (
        np.array(get_blob_energies(test_generator.inputs, setup)).T,
        test_generator.targets,
    )

    inference_model = setup["model"]["blob"]["inference_model"]

    model = train_blobs(train_features, train_labels, inference_model)
    logger.info(
        f"Accuracy on trainig set is: {model.score(train_features, train_labels)}"
    )
    logger.info(
        f"Accuracy on validation set is: {model.score(val_features, val_labels)}"
    )
    logger.info(f"Accuracy on test set is: {model.score(test_features, test_labels)}")

    # make_inference_plots(train_folder, test_features, test_labels)
    with open("/home/rmoretti/TESI/output_perf_blob/test/accuracy.txt", "a+") as f:
        f.write(str(model.score(test_features, test_labels)))
        f.write("\n")
        f.write(str(setup["detector"]["min_energy"]))
        f.write("\n")