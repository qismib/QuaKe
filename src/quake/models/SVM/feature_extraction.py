import logging
from ..CNN.CNN_dataloading import load_data
from ..CNN.CNN_data_preprocessing import prepare
from .load_CNN import load_cnn
from quake import PACKAGE

import numpy as np

logger = logging.getLogger(PACKAGE + ".SVM")


def extract_feats(data_folder, setup):
    """
    Extracts features from voxelised data and partitions into train, validation and test.

    Parameters
    ----------
        - data_folder: the input data folder path
        - setup: settings dictionary
    """

    opts = setup["model"]["cnn"]
    feat_size = opts["feature_number"]
    extra = setup["model"]["svm"]["extrafeats"]
    detector = setup["detector"]

    logger.info("Loading data and extracting features from CNN:")
    logger.info("Loading data ...")
    sig, bkg = load_data(data_folder, setup)

    data, labels = prepare(sig, bkg, setup)
    feature_layer, train_map, val_map, test_map = load_cnn(data_folder)

    feat_size = opts["feature_number"]

    # making batches for better memory allocation
    nbatches = 100
    data_feats = np.zeros((0, feat_size))
    for i in range(0, labels.shape[0], nbatches):
        input = [
            data[0][i : i + nbatches],
            data[1][i : i + nbatches],
            data[2][i : i + nbatches],
        ]
        data_feats = np.vstack((data_feats, feature_layer(input).numpy()))

    if extra:
        f = extrafeatures(data)
        data_feats = np.hstack((data_feats, f))

    set_train = data_feats[train_map]
    set_val = data_feats[val_map]
    set_test = data_feats[test_map]
    labels_train = labels[train_map]
    labels_val = labels[val_map]
    labels_test = labels[test_map]

    set_train, labels_train = outlier_removal(set_train, labels_train)

    set = [set_train, set_val, set_test]
    labels = [labels_train, labels_val, labels_test]

    return set, labels


def extrafeatures(data):
    """
    Adds customizable non-deep features

    Parameters
    ----------
        - data_folder: dataset with deep features
    """
    custom_feats = np.zeros((data[0].shape[0], 2))
    for i, particle in enumerate(data[0]):
        custom_feats[i, 0] = np.argwhere(particle).shape[0]
        custom_feats[i, 1] = particle[:, :, 0].sum()
    return custom_feats


def outlier_removal(sample_train, labels_train):
    """
    Removes outliers from the training set

    Parameters
    ----------
        - sample_train: train set partition
        - labels_train: train labels partition
    """
    means = np.mean(sample_train, axis=0)
    stds = np.std(sample_train, axis=0)
    outmask = (sample_train - means) / stds < 3.5  # Important to optimize
    idx = np.ones(labels_train.shape[0])
    for i in range(0, means.shape[0]):
        idx = np.logical_and(idx, outmask[:, i])
    return sample_train[idx], labels_train[idx]


""" Alternative version: eliminates outliers of the sig and bkg distributions separately.
def outlier_removal(sample_train, labels_train):

    means_sig = np.mean(sample_train[labels_train == 1], axis = 0)
    means_bkg = np.mean(sample_train[labels_train == 0], axis = 0)
    stds_sig = np.std(sample_train[labels_train == 1], axis=0)
    stds_bkg = np.std(sample_train[labels_train == 0], axis=0)

    means = np.zeros(sample_train.shape)
    stds = np.zeros(sample_train.shape)
    means[labels_train == 1] = means_sig
    means[labels_train == 0] = means_bkg
    stds[labels_train == 1] = stds_sig
    stds[labels_train == 0] = stds_bkg

    outmask = (sample_train - means) / stds < 3.5
    idx = np.ones(labels_train.shape[0])
    for i in range(0, means.shape[1]):
        idx = np.logical_and(idx, outmask[:, i])
    return sample_train[idx], labels_train[idx]
"""
