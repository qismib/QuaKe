from tensorflow import keras
from pathlib import Path
from tensorflow.keras import Model

import numpy as np


def load_cnn(data_folder: Path):
    try:
        model = keras.models.load_model(str(data_folder.parent) + "/models/cnn/CNN.h5")
        feature_layer = Model(
            inputs=model.inputs, outputs=model.get_layer("features").output
        )

        tr_map = np.loadtxt(str(data_folder.parent) + "/models/cnn/train_map").astype(
            int
        )
        val_map = np.loadtxt(
            str(data_folder.parent) + "/models/cnn/validation_map"
        ).astype(int)
        te_map = np.loadtxt(str(data_folder.parent) + "/models/cnn/test_map").astype(
            int
        )
    except:
        raise Exception(
            "SVM needs features computed by the CNN model, but there is no model "
            f"in folder {data_folder.parent / 'models/cnn'} "
            "Please run 'train' with '--model cnn' first and try again."
        )

    return feature_layer, tr_map, val_map, te_map
