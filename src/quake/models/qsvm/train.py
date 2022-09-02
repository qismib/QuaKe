""" This module provides functions for SVM training and from feature extraction."""
import logging
from pathlib import Path
import pickle
from typing import Tuple
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.metrics import roc_auc_score
from ..svm.utils import extract_feats, rearrange_scale
from ..attention.train import load_and_compile_network as load_attention_network
from ..attention.attention_dataloading import read_data as read_data_attention
from ..cnn.cnn_dataloading import read_data as read_data_cnn
from ..cnn.train import load_and_compile_network as load_cnn_network
from quake import PACKAGE
from quake.dataset.generate_utils import Geometry
from .quantum_featuremaps import (
    custom2_featuremap,
    genetic_featuremap,
    genetic_featuremap_2,
)
from .qsvm_tester import make_kernels
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator

logger = logging.getLogger(PACKAGE + ".qsvm")


def qsvm_hyperparameter_training(
    train_folder: Path,
    dataset: list[np.ndarray],
    labels: list[np.ndarray],
    qt_kernels: list([QuantumCircuit]),
    kernel_titles: list([str]),
    should_add_extra_feats: bool,
) -> list[SVC]:
    """Grid-search optimization of QSVM hyperparameters.

    Parameters
    ----------
    train_folder: Path
        The train output folder path.
    dataset: list[np.ndarray]
        Partitioned dataset [train, validation, test].
    labels: list[np.ndarray]
        Partitioned labels [train, validation, test].
    qt_kernels: list([QuantumCircuit])
        Quantum kernels.
    kernel_titles: list(str)
        Quantum kernel titles.
    should_add_extra_feats: bool
        Wether to enhance extracted features with custom ones.

    Returns
    -------
    models: list[SVC]
        The trained support vector classifiers. One for each classical kernel.
    """
    logger.info("Training, validating, testing QSVMs ...")
    feature_size = dataset[0].shape[1]
    if should_add_extra_feats:
        logger.info(
            f"Using {feature_size} features extracted from CNN "
            "+ Total event energy + Nhits"
        )
    else:
        logger.info(f"Using {feature_size} features extracted from CNN")

    quantum_grid = {"C": [0.01, 0.1, 1, 10, 100, 1000, 10000]}

    train_size = labels[0].shape[0]
    # val_size = labels[1].shape[0]
    val_size = 100  # for saving computing time
    partitions = np.append(
        -np.ones(train_size, dtype=int),
        np.zeros(val_size, dtype=int),
    )
    validation_idx = PredefinedSplit(partitions)

    set_train_val = np.concatenate((dataset[0], dataset[1][:val_size]), axis=0)
    labels_train_val = np.concatenate((labels[0], labels[1][:val_size]), axis=0)

    models = []

    for k, kernel in enumerate(qt_kernels):
        logger.info(f"Fitting QSVC with {kernel_titles[k]} kernel")
        grid = GridSearchCV(
            SVC(kernel="precomputed", probability=True),
            quantum_grid,
            refit=True,
            verbose=3,
            cv=validation_idx,
        )
        ker_matrix_cv = kernel.evaluate(x_vec=set_train_val)
        grid.fit(ker_matrix_cv, labels_train_val)
        logger.info(grid.best_params_)
        quantum_svc = SVC(probability=True, **grid.best_params_)
        ker_matrix_train = ker_matrix_cv[:train_size, :train_size]
        quantum_svc.fit(ker_matrix_train, labels[0])
        pickle.dump(
            quantum_svc,
            open(train_folder / Path(kernel_titles[k] + ".sav"), "wb"),
        )
        models.append(quantum_svc)
    return models


def evaluate_svm(
    models: list[SVC],
    dataset: list[np.ndarray],
    labels: list[np.ndarray],
    qt_kernels: QuantumCircuit,
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
        List of trained QSVMs with different kernels.
    dataset:
        List of arrays with extracted features [train, validation, test]. Each
        array has shape=(nb events, nb features).
    labels:
        List of arrays with labels [train, validation, test]. Each
        array has shape=(nb events,).
    qt_kernels: QuantumCircuit
        Quantum kernels.
    kernel_titles: list[str]
        Quantum kernel names.
    """

    accuracy = lambda y, l: np.sum(y == l) / l.shape[0]
    sensitivity = lambda y, l: np.sum(np.logical_and(y == 1, l == 1)) / np.sum(l)
    specificity = lambda y, l: np.sum(np.logical_and(y == 0, l == 0)) / np.sum(
        np.logical_not(l)
    )
    k_len = len(qt_kernels)

    acc = np.zeros((k_len, 3))
    sen = np.zeros((k_len, 3))
    spec = np.zeros((k_len, 3))
    auc = np.zeros((k_len, 3))

    for k, qsvc in enumerate(models):
        for j in range(0, 3):
            if j == 0:
                ker_matrix = qt_kernels[k].evaluate(x_vec=dataset[j])
            else:
                ker_matrix = qt_kernels[k].evaluate(x_vec=dataset[j], y_vec=dataset[0])
            y = qsvc.predict(ker_matrix)
            acc[k, j] = accuracy(y, labels[j])
            sen[k, j] = sensitivity(y, labels[j])
            spec[k, j] = specificity(y, labels[j])
            y_prob = models[k].predict_proba(ker_matrix)[:, 1]
            auc[k, j] = roc_auc_score(labels[j], y_prob)
    logger.info("Metrics matrices. Columns: train, validation, test")
    np.set_printoptions(precision=3)
    logger.info("Accuracy: \n" f"{acc}")
    logger.info("Sensitivity: \n" f"{sen}")
    logger.info("Specificity: \n" f"{spec}")
    logger.info("AUC: \n" f"{auc}")


def load_quantum_kernels(qubits: int) -> Tuple[QuantumCircuit, str]:
    """Prepares the quantum kernel objects from some featuremaps

    Parameters
    ----------
    qubits: int
        Number of qubits, equivalent to the number of features of the dataset.

    Returns
    -------
    """
    x = ParameterVector("x", length=qubits)
    backend = AerSimulator(method="statevector")
    c2 = custom2_featuremap(x, qubits)
    gen1 = genetic_featuremap(x)
    gen2 = genetic_featuremap_2(x)
    qt_kernels = make_kernels([c2, gen1, gen2], backend)
    kernel_titles = ["c2", "gen1", "gen2"]
    return qt_kernels, kernel_titles


def qsvm_train(data_folder: Path, train_folder: Path, setup: dict):
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
    extractor_type = setup["model"]["qsvm"]["feature_extractor"].lower()
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
    should_add_extra_feats = setup["model"]["qsvm"]["should_add_extra_feats"]
    split_ratio = setup["model"]["qsvm"]["split_ratio"]
    train_features, train_labels = extract_feats(
        train_generator, network, should_add_extra_feats, should_remove_outliers=True
    )
    val_features, val_labels = extract_feats(
        val_generator, network, should_add_extra_feats, should_remove_outliers=False
    )
    test_features, test_labels = extract_feats(
        test_generator, network, should_add_extra_feats, should_remove_outliers=False
    )

    dataset = rearrange_scale(
        train_features,
        val_features,
        test_features,
        setup["model"]["qsvm"]["should_do_scaling"],
    )
    labels = [train_labels, val_labels, test_labels]
    dataset[0], labels[0] = train_test_split(
        dataset[0], labels[0], train_size=split_ratio, random_state=42
    )[::2]

    qubits = dataset[0].shape[1]
    quantum_kernels, kernel_titles = load_quantum_kernels(qubits)

    # SVM training and hyperparameter optimization
    quantum_svms = qsvm_hyperparameter_training(
        train_folder,
        dataset,
        labels,
        quantum_kernels,
        kernel_titles,
        should_add_extra_feats,
    )

    # evaluating the performances on train, validation and test sets
    evaluate_svm(
        quantum_svms,
        dataset,
        labels,
        quantum_kernels,
    )
