import logging
import numpy as np
from sklearn.model_selection import train_test_split
from quake import PACKAGE

logger = logging.getLogger(PACKAGE)


def roll_crop(plane, crop):
    """
    Shifts tracks to a corner and reduces dimensionality by cropping

    Parameters
    ----------
        - plane: 2D projections
        - crop: numbers of bins to keep
    """
    for i, particle in enumerate(plane):
        c = np.argwhere(particle)
        if list(c):
            cm1 = np.min(c[:, 0])
            cm2 = np.min(c[:, 1])
            plane[i] = np.roll(plane[i], -cm1, axis=0)
            plane[i] = np.roll(plane[i], -cm2, axis=1)
    plane = plane[:, 0 : crop[0], 0 : crop[1]]
    return plane


def prepare(sig, bkg, setup):
    """
    Returns processed data. Dimensionality reduction for optimization

    Parameters
    ----------
        - sig: 2D projections of signals
        - bkg: 2D projections of background
        - setup: settings dictionary
    """
    logger.info("Preparing data ...")
    res = np.array(setup["detector"]["resolution"])
    YZ_plane = np.concatenate((sig[0], bkg[0]), axis=0)
    XZ_plane = np.concatenate((sig[1], bkg[1]), axis=0)
    XY_plane = np.concatenate((sig[2], bkg[2]), axis=0)
    hist_crop = (20 / res).astype(int)
    YZ_plane = roll_crop(YZ_plane, hist_crop[[1, 2]])
    XZ_plane = roll_crop(XZ_plane, hist_crop[[0, 2]])
    XY_plane = roll_crop(XY_plane, hist_crop[[0, 1]])

    nsig = sig[0].shape[0]
    nbkg = bkg[0].shape[0]

    labels = np.concatenate([np.ones(nsig), np.zeros(nbkg)])

    data = [YZ_plane, XZ_plane, XY_plane]

    return data, labels


def tr_val_te_split(data, labels, setup, data_folder):
    """
    Returns dataset partitioning into train, validation and test sets, in this format:
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
        - data: 2D projections dataset
        - labels: class labels
        - setup: settings dictionary
        - data_folder: the input data folder path
    """
    test_size = setup["model"]["cnn"]["val_te_ratio"]
    seed = setup["seed"]
    YZ_plane = data[0]
    XZ_plane = data[1]
    XY_plane = data[2]

    logger.info(YZ_plane.shape)

    ntot = YZ_plane.shape[0]
    idx = np.arange(0, ntot)

    YZ_plane_idx = np.vstack((YZ_plane.reshape(ntot, -1).T, idx)).T

    YZ_train, YZ_val_test, labels_train, labels_val_test = train_test_split(
        YZ_plane_idx, labels, test_size=2 * test_size, random_state=seed
    )
    YZ_val, YZ_te, labels_val, labels_test = train_test_split(
        YZ_val_test, labels_val_test, test_size=0.5, random_state=seed
    )

    idx = [
        YZ_train[:, -1].astype(int),
        YZ_val[:, -1].astype(int),
        YZ_te[:, -1].astype(int),
    ]

    set = []
    for id in idx:
        set.append([YZ_plane[id], XZ_plane[id], XY_plane[id]])
    labels = [labels_train, labels_val, labels_test]

    logger.info("Saving train-validation-test partition for the SVM model")
    np.savetxt(str(data_folder.parent) + "/models/cnn/train_map", idx[0], fmt="%i")
    np.savetxt(str(data_folder.parent) + "/models/cnn/validation_map", idx[1], fmt="%i")
    np.savetxt(str(data_folder.parent) + "/models/cnn/test_map", idx[2], fmt="%i")

    return set, labels


def display_dataset_partition(label_train, labels):
    """
    Displays the dataset partition.

    Parameters
    ----------
        - label_train: labels in use for training over the total training set
        - labels: labels of partitioned dataset [train, validation, test]
    """
    frac = lambda l, n: np.sum(l == n) / len(l)
    ntot = len(labels[0]) + len(labels[1]) + len(labels[2])

    logger.info(
        f"{len(label_train)} samples for train ("
        f"{len(label_train)/ntot:.1%} of total) ("
        f"{frac(label_train, 1):.1%} signal, "
        f"{frac(label_train, 0):.1%} background)"
    )
    logger.info(
        f"{len(labels[1])} samples for validation ("
        f"{len(labels[1])/ntot:.1%} of total) ("
        f"{frac(labels[1], 1):.1%} signal, "
        f"{frac(labels[1], 0):.1%} background)"
    )
    logger.info(
        f"{len(labels[2])} samples for test ("
        f"{len(labels[2])/ntot:.1%} of total) ("
        f"{frac(labels[2], 1):.1%} signal, "
        f"{frac(labels[2], 0):.1%} background)"
    )
