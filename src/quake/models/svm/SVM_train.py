import logging
from pathlib import Path
import pickle
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.metrics import roc_auc_score
from quake import PACKAGE

logger = logging.getLogger(PACKAGE + ".SVM")


def svm_hyperparameter_training(
    train_folder: Path,
    dataset: list[np.ndarray],
    labels: list[np.ndarray],
    kernels: list[str],
    should_add_extra_feats: bool,
    should_do_scaling: bool,
) -> list[SVC]:
    """Grid-search optimization of SVM the hyperparameters.

    Parameters
    ----------
    train_folder: Path
        The train output folder path.
    dataset: list[np.ndarray]
        Partitioned dataset [train, validation, test]
    labels: list[np.ndarray]
        Partitioned labels [train, validation, test]
    kernels: list[str]
        Classical SVM kernels labels.
    should_add_extra_feats: bool
        Wether to enhance extracted features with custom ones.
    should_do_scaling: bool
        Wether to do input scaling or not.

    Returns
    -------
    models: list[SVC]
        The trained support vector classifiers. One for each classical kernel.
    """
    logger.info("Training, validating, testing SVMs with linear, poly, rbf kernels ...")
    feature_size = dataset[0].shape[1]
    if should_add_extra_feats:
        logger.info(
            f"Using {feature_size} features extracted from CNN "
            "+ Total event energy + Nhits"
        )
    else:
        logger.info(f"Using {feature_size} features extracted from CNN")

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

    # TODO: inspect further this split
    # train on just a subset of events
    split_ratio = 0.1
    set_train_svm, labels_train_svm = train_test_split(
        dataset[0], labels[0], train_size=split_ratio, random_state=42
    )[::2]

    grids = [linear_grid, poly_grid, rbf_grid]

    partitions = np.append(
        -np.ones(labels_train_svm.shape[0], dtype=int),
        np.zeros(labels[1].shape[0], dtype=int),
    )
    validation_idx = PredefinedSplit(partitions)

    set_train_val = np.concatenate((set_train_svm, dataset[1]), axis=0)
    labels_train_val = np.concatenate((labels_train_svm, labels[1]), axis=0)

    models = []

    for k, kernel in enumerate(kernels):
        logger.info(f"Fitting SVC with {kernel} kernel")
        grid = GridSearchCV(SVC(), grids[k], refit=True, verbose=0, cv=validation_idx)
        scaled_cv = scaler(set_train_val, kernel, should_do_scaling)
        scaled_s_tr_svm = scaler(set_train_svm, kernel, should_do_scaling)
        grid.fit(scaled_cv, labels_train_val)
        classical_svc = SVC(probability=True, **grid.best_params_)
        classical_svc.fit(scaled_s_tr_svm, labels_train_svm)
        pickle.dump(
            classical_svc,
            open(train_folder / "{kernel}.sav", "wb"),
        )
        models.append(classical_svc)
    return models


def evaluate_svm(
    models: list[SVC],
    dataset: list[np.ndarray],
    labels: list[np.ndarray],
    kernels: list[str],
    should_do_scaling: bool,
):
    """Evaluates performance scores.

    Metrics computed for every available kernel separately on train, validation
    and test sets:
    - accuracy
    - sensitivity
    - specificity
    - AUC

    Parameters
    ----------
    models: list[SVC]
        List of trained SVMs with different kernels.
    dataset:
        List of arrays with extracted features [train, validation, test]. Each
        array has shape=(nb events, nb features).
    labels:
        List of arrays with labels [train, validation, test]. Each
        array has shape=(nb events,).
    kernels: list[str]
        Classical SVM kernels labels.
    should_do_scaling: bool
        Wether to do input scaling or not.
    """
    accuracy = lambda y, l: np.sum(y == l) / l.shape[0]
    sensitivity = lambda y, l: np.sum(np.logical_and(y == 1, l == 1)) / np.sum(l)
    specificity = lambda y, l: np.sum(np.logical_and(y == 0, l == 0)) / np.sum(
        np.logical_not(l)
    )
    k_len = len(kernels)

    acc = np.zeros((k_len, 3))
    sen = np.zeros((k_len, 3))
    spec = np.zeros((k_len, 3))
    auc = np.zeros((k_len, 3))

    for k, kernel in enumerate(kernels):
        for j in range(0, 3):
            scaled_set = scaler(dataset[j], kernel, should_do_scaling)
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
