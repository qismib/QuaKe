import logging
from pathlib import Path
from tensorflow.keras.utils import to_categorical
import numpy as np

from quake import PACKAGE
from .CNN_dataloading import load_data
from .CNN_data_preprocessing import prepare, tr_val_te_split, display_dataset_partition
from .CNN_Net import buildCNN
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(PACKAGE + ".CNN")


def train(model, set, labels, setup, data_folder):
    """
    Trains the CNN
    set must be as follows:
    set[idx1][idx2]
      idx1:
          = 0 train
          = 1 validation
          = 2 test
      idx2:
          = 0 YZ plane
          = 1 XZ plane
          = 2 XY plane
    Parameters
    ----------
        - model: the CNN model
        - set: dataset: 2D projections of voxelized data
        - labels: class labels
        - setup: settings dictionary
        - data_folder: the input data folder path
    """
    logger.info("Training the CNN")

    batch_size = setup["model"]["cnn"]["batch_size"]
    epochs = setup["model"]["cnn"]["epochs"]
    model.fit(
        [set[0][0], set[0][1], set[0][2]],
        to_categorical(labels[0]),
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([set[1][0], set[1][1], set[1][2]], to_categorical(labels[1])),
        verbose=1,
    )

    logger.info("Saving model as 'CNN.h5'")
    model.save(str(data_folder.parent) + "/models/cnn/CNN.h5")
    return model


def evaluate_CNN(model, set_test, labels_test):
    """
    Evaluates accuracy, sensitivity, specificity, AUC for the test set

    Parameters
    ----------
        - model: the CNN model
        - set_test: set partition for testing
        - labels_test: label partition for testing
    """
    accuracy = lambda y, l: np.sum(y == l) / l.shape[0]
    sensitivity = lambda y, l: np.sum(np.logical_and(y == 1, l == 1)) / np.sum(l)
    specificity = lambda y, l: np.sum(np.logical_and(y == 0, l == 0)) / np.sum(
        np.logical_not(l)
    )

    metrics = ["Accuracy", "Sensitivity", "Specificity", "auc"]

    scores = model.predict([set_test[0], set_test[1], set_test[2]])
    y_prob = scores[:, 1]
    y = np.argmax(scores, axis=1)
    results = [
        accuracy(y, labels_test),
        sensitivity(y, labels_test),
        specificity(y, labels_test),
        roc_auc_score(labels_test, y_prob),
    ]
    logger.info("Performances on test set:")
    for i, m in enumerate(metrics):
        logger.info(f"{m}" f"{results[i]:{10}.{4}}")


def CNN_train(data_folder: Path, setup):
    """
    CNN training.

    Parameters
    ----------
        - data_folder: the input data folder path
        - setup: settings dictionary
    """
    # dataset loading
    sig, bkg = load_data(data_folder, setup)

    # preprocessing
    data, labels = prepare(sig, bkg, setup)

    # dataset partitioning into train, validation and test sets
    set, labels = tr_val_te_split(data, labels, setup, data_folder)

    # displaying dataset partitioning
    display_dataset_partition(labels[0], labels)

    # defining CNN architecture
    model = buildCNN(set, setup)

    # training and saving the CNN
    model = train(model, set, labels, setup, data_folder)

    # evaluating the performances on train, validation and test sets
    evaluate_CNN(model, set[2], labels[2])
