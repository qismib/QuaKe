"""
    This module implements function for data loading for CNN.
"""

import logging
from quake import PACKAGE
from pathlib import Path
import numpy as np
import scipy
import tensorflow as tf


logger = logging.getLogger(PACKAGE)


def get_data(file: Path, size):
    """
    Returns 2D projections from voxelized data

    Parameters
    ----------
        - file: filename of the '.npz' file
        - size: 2D projections' dimension
    """
    matrix = scipy.sparse.load_npz(file)
    matrix = matrix.tocoo()
    indices = np.mat([matrix.row, matrix.col]).transpose()
    matrix = tf.SparseTensor(indices, matrix.data, matrix.shape)
    matrix = tf.sparse.reshape(matrix, (-1, size[0], size[1], size[2]))
    YZ_plane = np.expand_dims(tf.sparse.reduce_sum(matrix, axis=1), 3)
    XZ_plane = np.expand_dims(tf.sparse.reduce_sum(matrix, axis=2), 3)
    XY_plane = np.expand_dims(tf.sparse.reduce_sum(matrix, axis=3), 3)
    return YZ_plane, XZ_plane, XY_plane


def load_data(folder: Path, setup):
    """
    Returns 2D projections from the voxelized data folder

    Parameters
    ----------
        - folder: path of the '.npz' files
        - setup: settings dictionary
    """
    logger.info("Loading data ...")
    res = np.array(setup["detector"]["resolution"])
    size = np.ceil(40 / res).astype(int)

    data_sig_x = []
    data_sig_y = []
    data_sig_z = []

    data_bkg_x = []
    data_bkg_y = []
    data_bkg_z = []

    for file in folder.iterdir():
        logger.info("Loading" + str(file))

        # signal filenames starts with "b"
        is_signal = file.name[0] == "b"
        # background filenames starts with "e"
        is_background = file.name[0] == "e"
        if file.suffix == ".npz":
            YZ_plane, XZ_plane, XY_plane = get_data(file, size)
            if is_signal:
                data_sig_x.append(YZ_plane)
                data_sig_y.append(XZ_plane)
                data_sig_z.append(XY_plane)
            elif is_background:
                data_bkg_x.append(YZ_plane)
                data_bkg_y.append(XZ_plane)
                data_bkg_z.append(XY_plane)

    data_sig_x = np.concatenate(data_sig_x, axis=0)
    data_sig_y = np.concatenate(data_sig_y, axis=0)
    data_sig_z = np.concatenate(data_sig_z, axis=0)

    data_bkg_x = np.concatenate(data_bkg_x, axis=0)
    data_bkg_y = np.concatenate(data_bkg_y, axis=0)
    data_bkg_z = np.concatenate(data_bkg_z, axis=0)

    data_sig = [data_sig_x, data_sig_y, data_sig_z]
    data_bkg = [data_bkg_x, data_bkg_y, data_bkg_z]

    return data_sig, data_bkg
