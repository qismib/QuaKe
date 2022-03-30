"""
    This module implements function for data loading for CNN.
"""

import logging
from quake import PACKAGE
from pathlib import Path
import numpy as np
import scipy
import tensorflow as tf


logger = logging.getLogger(PACKAGE + ".attention")


def get_data(file: Path, size):
    m = scipy.sparse.load_npz(file)
    m = m.tocoo()
    indices = np.mat([m.row, m.col]).transpose()
    m = tf.SparseTensor(indices, m.data, m.shape)
    m = tf.sparse.reshape(m, (-1, size[0], size[1], size[2]))
    dx = np.expand_dims(tf.sparse.reduce_sum(m, axis=1), 3)
    dy = np.expand_dims(tf.sparse.reduce_sum(m, axis=2), 3)
    dz = np.expand_dims(tf.sparse.reduce_sum(m, axis=3), 3)
    return dx, dy, dz


def load_data(folder: Path, detector):
    res = np.array(detector["resolution"])
    size = np.ceil(40 / res).astype(int)

    data_sig_x = []
    data_sig_y = []
    data_sig_z = []

    data_bkg_x = []
    data_bkg_y = []
    data_bkg_z = []

    for file in folder.iterdir():
        logger.info("Loading" + str(file))

        # signal files
        is_signal = file.name[0] == "b"
        is_background = file.name[0] == "e"
        if file.suffix == ".npz":
            dx, dy, dz = get_data(file, size)
            if is_signal:
                data_sig_x.append(dx)
                data_sig_y.append(dy)
                data_sig_z.append(dz)
            # background files
            elif is_background:
                data_bkg_x.append(dx)
                data_bkg_y.append(dy)
                data_bkg_z.append(dz)

    data_sig_x = np.concatenate(data_sig_x, axis=0)
    data_sig_y = np.concatenate(data_sig_y, axis=0)
    data_sig_z = np.concatenate(data_sig_z, axis=0)

    data_bkg_x = np.concatenate(data_bkg_x, axis=0)
    data_bkg_y = np.concatenate(data_bkg_y, axis=0)
    data_bkg_z = np.concatenate(data_bkg_z, axis=0)

    data_sig = [data_sig_x, data_sig_y, data_sig_z]
    data_bkg = [data_bkg_x, data_bkg_y, data_bkg_z]

    return data_sig, data_bkg
