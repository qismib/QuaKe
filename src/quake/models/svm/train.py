""" This module provides functions for SVM training and from feature extraction."""
import logging
from pathlib import Path
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.metrics import roc_auc_score
from .utils import extract_feats, scaler
from ..attention.train import load_and_compile_network as load_attention_network
from ..attention.attention_dataloading import read_data as read_data_attention
from ..cnn.cnn_dataloading import read_data as read_data_cnn
from ..cnn.train import load_and_compile_network as load_cnn_network
from quake import PACKAGE
from quake.dataset.generate_utils import Geometry

logger = logging.getLogger(PACKAGE + ".svm")


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


def svm_train(data_folder: Path, train_folder: Path, setup: dict):
    """SVM training.

    Parameters
    ----------
    data_folder: Path
        The input data folder path.
    train_folder: Path
        The train output folder path.
    setup: dict
        Settings dictionary.

    Raises
    ------
    NotImplementedError
        If extractor type not one of `svm` or `attention`
    """
    extractor_type = setup["model"]["svm"]["feature_extractor"].lower()
    load_map_folder = train_folder.parent / extractor_type

    if extractor_type == "cnn":
        read_data_fn = read_data_cnn
        load_net_fn = load_cnn_network
    elif extractor_type == "attention":
        read_data_fn = read_data_attention
        load_net_fn = load_attention_network
    else:
        raise NotImplementedError(
            f"exctractor model not implemented, found: {extractor_type}"
        )

    train_generator, val_generator, test_generator = read_data_fn(
        data_folder, load_map_folder, setup, split_from_maps=True
    )

    geo = Geometry(setup["detector"])
    # extractor setup
    esetup = setup["model"][extractor_type]
    esetup.update({"ckpt": load_map_folder / f"{extractor_type}.h5"})
    network = load_net_fn(esetup, setup["run_tf_eagerly"], geo=geo)
    should_add_extra_feats = setup["model"]["svm"]["should_add_extra_feats"]
    train_features, train_labels = extract_feats(
        train_generator, network, should_add_extra_feats
    )
    val_features, val_labels = extract_feats(
        val_generator, network, should_add_extra_feats
    )
    test_features, test_labels = extract_feats(
        test_generator, network, should_add_extra_feats
    )

    # training and saving the SVMs
    dataset = [train_features, val_features, test_features]
    labels = [train_labels, val_labels, test_labels]

    print(list(map(lambda x: x.shape, dataset)))
    print(list(map(lambda x: x.shape, labels)))
    exit()

    # SVM training and hyperparameter optimization
    classical_svms = svm_hyperparameter_training(
        train_folder,
        dataset,
        labels,
        setup["model"]["svm"]["kernels"],
        should_add_extra_feats,
        setup["model"]["svm"]["should_do_scaling"],
    )

    # evaluating the performances on train, validation and test sets
    evaluate_svm(
        classical_svms,
        dataset,
        labels,
        setup["model"]["svm"]["kernels"],
        setup["model"]["svm"]["should_do_scaling"],
    )
