import tensorflow as tf
from pathlib import Path
from tensorflow.keras import Model

import numpy as np


def load_cnn(data_folder: Path):
    """
    Returns a previously trained CNN model and train-validation-test masks

    Parameters
    ----------
        - data_folder: the input data folder path

    Returns
    -------
    tf.keras.Model
        Loaded CNN model

    """
    model_path = data_folder.parent / "models/cnn"
    try:
        model = tf.keras.models.load_model(model_path / "CNN.h5")
        feature_layer = Model(
            inputs=model.inputs, outputs=model.get_layer("features").output
        )

        train_map = np.loadtxt(model_path / "train_map").astype(int)
        val_map = np.loadtxt(model_path / "validation_map").astype(int)
        test_map = np.loadtxt(model_path / "test_map").astype(int)
    except:
        raise Exception(
            "SVM needs features computed by the CNN model, but there is no model "
            f"in folder {model_path} "
            "Please run 'train' with '--model cnn' first and try again."
        )

    return feature_layer, train_map, val_map, test_map
