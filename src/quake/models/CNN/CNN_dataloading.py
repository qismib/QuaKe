"""
    This module implements function for data loading for CNN.
"""

import logging
from quake import PACKAGE
from pathlib import Path
import numpy as np
import scipy


logger = logging.getLogger(PACKAGE + ".attention")


def get_data(file: Path):
    matrix = scipy.sparse.load_npz(file).todense()
    return np.array(matrix).reshape(1000, 8, 8, 40)


def load_data(folder: Path):
    data_sig_l = []
    data_bkg_l = []
    for file in folder.iterdir():
        # signal files
        is_signal = file.name[0] == "b"
        is_background = file.name[0] == "e"
        if is_signal:
            if file.suffix == ".npz":
                data_sig_l.append(get_data(file))
        # background files
        elif is_background:
            if file.suffix == ".npz":
                data_bkg_l.append(get_data(file))

    data_sig_l = np.concatenate(data_sig_l, axis=0)
    data_bkg_l = np.concatenate(data_bkg_l, axis=0)

    return data_sig_l, data_bkg_l
