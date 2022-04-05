import logging
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from quake import PACKAGE
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn import preprocessing

from .feature_extraction import extract_feats
from ..CNN.CNN_train import display_dataset_partition

logger = logging.getLogger(PACKAGE + ".SVM")

classical_kernels = ["linear", "poly", "rbf"]
do_scaling = False


def train_optimize(set, labels, set_train_svm, labels_train_svm, extra, data_folder):
    """
    Grid-search optimizes the hyperparameters and trains the SVMs

    Parameters
    ----------
        - set: partitioned dataset [train, validation, test]
        - labels: partitioned labels [train, validation, test]
        - set_train_svm: set in use for SVMs trainings
        - labels_train_svm: labels in use for SVMs trainings
        - extra: if true, non-deep features are included in the dataset
        - data_folder: the input data folder path
    """
    logger.info("Training, validating, testing SVMs with linear, poly, rbf kernels ...")
    feature_size = set[0].shape[1]
    if extra:
        logger.info(
            "Using "
            f"{feature_size} features extracted from CNN + Total event energy + Nhits"
        )
    else:
        logger.info("Using " f"{feature_size} features extracted from CNN")

    linear_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "gamma": [10, 1, 0.1, 0.01, 0.001],
        "kernel": ["linear"],
    }
    poly_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "kernel": ["poly"],
        "degree": [2, 3],
    }
    rbf_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "gamma": [10, 1, 0.1, 0.01, 0.001],
        "kernel": ["rbf"],
    }

    grids = [linear_grid, poly_grid, rbf_grid]

    partitions = np.append(
        -np.ones(labels_train_svm.shape[0], dtype=int),
        np.zeros(labels[1].shape[0], dtype=int),
    )
    validation_idx = PredefinedSplit(partitions)

    set_train_val = np.concatenate((set_train_svm, set[1]), axis=0)
    labels_train_val = np.concatenate((labels_train_svm, labels[1]), axis=0)

    models = []

    for k, kernel in enumerate(classical_kernels):
        grid = GridSearchCV(SVC(), grids[k], refit=True, verbose=0, cv=validation_idx)
        scaled_cv = scaler(set_train_val, kernel, do_scaling)
        scaled_s_tr_svm = scaler(set_train_svm, kernel, do_scaling)
        grid.fit(scaled_cv, labels_train_val)
        classical_svc = SVC(probability=True, **grid.best_params_)
        classical_svc.fit(scaled_s_tr_svm, labels_train_svm)
        pickle.dump(
            classical_svc,
            open(str(data_folder.parent) + "/models/svm/" + kernel + ".sav", "wb"),
        )
        models.append(classical_svc)
    return models


def evaluate_SVM(models, set, labels):
    """
    Evaluates accuracy, sensitivity, specificity, AUC for every kernel on train, validation and test sets

    Parameters
    ----------
        - models: list of trained SVMs with different kernels
        - set: partitioned dataset [train, validation, test]
        - labels: partitioned labels [train, validation, test]
    """
    accuracy = lambda y, l: np.sum(y == l) / l.shape[0]
    sensitivity = lambda y, l: np.sum(np.logical_and(y == 1, l == 1)) / np.sum(l)
    specificity = lambda y, l: np.sum(np.logical_and(y == 0, l == 0)) / np.sum(
        np.logical_not(l)
    )
    k_len = len(classical_kernels)

    acc = np.zeros((k_len, 3))
    sen = np.zeros((k_len, 3))
    spec = np.zeros((k_len, 3))
    auc = np.zeros((k_len, 3))

    for k, kernel in enumerate(classical_kernels):
        for j in range(0, 3):
            scaled_set = scaler(set[j], kernel, do_scaling)
            y = models[k].predict(scaled_set)
            acc[k, j] = accuracy(y, labels[j])
            sen[k, j] = sensitivity(y, labels[j])
            spec[k, j] = specificity(y, labels[j])
            y_prob = models[k].predict_proba(scaled_set)[:, 1]
            auc[k, j] = roc_auc_score(labels[j], y_prob)
    logger.info(
        "Metrics matrices. Rows: linear, poly, rbf. Columns: train, validation, test"
    )
    np.set_printoptions(precision=3)
    logger.info("Accuracy: \n" f"{acc}")
    logger.info("Sensitivity: \n" f"{sen}")
    logger.info("Specificity: \n" f"{spec}")
    logger.info("AUC: \n" f"{auc}")


def scaler(set, kernel, do_scaling):
    """
    Transforms the input data for enhancing SVMs performances.

    Parameters
    ----------
        - set: partitioned dataset [train, validation, test]
        - kernel: kernel label
        - do_scaling: scaling toggle on/off
    """
    # #scaler = preprocessing.RobustScaler().fit(s_tr) # mean 0 std 1
    # scaler = MinMaxScaler()

    if do_scaling:
        if kernel == "linear" or kernel == "rbf":
            scaler = preprocessing.PowerTransformer(
                standardize=True
            )  # nonlinear map to gaussian
        elif kernel == "poly":
            scaler = preprocessing.QuantileTransformer(
                random_state=0
            )  # nonlinear map to uniform dist
        return scaler.fit_transform(set)
    else:
        return set


def SVM_train(data_folder: Path, setup):
    """
    Classical SVM training.

    Parameters
    ----------
        - data_folder: the input data folder path
        - setup: settings dictionary
    """
    extra = setup["model"]["svm"]["extrafeats"]

    # data loading and preprocessing
    set, labels = extract_feats(data_folder, setup)

    # training on a subset of the samples avaliable for training
    set_train_svm, labels_train_svm = train_test_split(
        set[0], labels[0], train_size=0.1, random_state=42
    )[::2]

    # displaying dataset partitioning
    display_dataset_partition(labels_train_svm, labels)

    # training and saving the SVMs
    classical_SVMs = train_optimize(
        set, labels, set_train_svm, labels_train_svm, extra, data_folder
    )

    # evaluating the performances on train, validation and test sets
    evaluate_SVM(classical_SVMs, set, labels)
