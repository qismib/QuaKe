"""
    This module implements function for data loading for AttentionNetwork.
    Dynamic batching is employed.
"""
import logging
from pathlib import Path
from math import ceil
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
import tensorflow as tf
from quake import PACKAGE
from quake.dataset.generate_utils import Geometry
from quake.utils.configflow import float_me

logger = logging.getLogger(PACKAGE + ".attention")

to_np = lambda x: np.array(x, dtype=object)


def padding(array: np.ndarray) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Pads the inputs to fit data into a tensor.

    Parameters
    ----------
        - np.ndarray, array of objects each of shape=([nb hits], nb features)

    Returns
    -------
        - tf.Tensor, padded data
        - tf.Tensor, mask
    """
    maxlen = array[-1].shape[0]

    pwidth = lambda x: [[0, maxlen - len(x)], [0, 0]]
    rows = [np.pad(row, pwidth(row), "constant", constant_values=0.0) for row in array]
    rows = np.stack(rows, axis=0)

    # TODO [enhancement]: the attention can be weighted according to hit relative distance
    pmask = lambda x: [[0, maxlen - len(x)]] * 2
    mask = lambda x: np.ones([len(x)] * 2)
    masks = [
        np.pad(mask(row), pmask(row), "constant", constant_values=0.0) for row in array
    ]
    masks = np.stack(masks, axis=0)

    return float_me(rows), float_me(masks)


class Dataset(tf.keras.utils.Sequence):
    """Dataset sequence."""

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 12345,
    ):
        """
        Parameters
        ----------
            - inputs: array of objects each of shape=([nb hits], nb features)
            - targets: np.ndarray, of shape=(nb events)
            - batch_size: int
            - shuffle: bool, wether to shuffle dataset on epoch end
            - seed: int, random generator seed for reproducibility

        """
        self.inputs = to_np(inputs)
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.default_rng(self.seed) if self.shuffle else None
        self.perm = np.arange(self.__len__())
        self.sort_data()

    def sort_data(self):
        """
        Sorts inputs and targets according to increasing number of hits in
        event. This is needed for dynamic batching.
        """
        fn = lambda pair: len(pair[0])
        indices = np.arange(len(self.inputs))
        idx = [i for _, i in sorted(zip(self.inputs, indices), key=fn)]
        self.inputs = self.inputs[idx]
        self.targets = self.targets[idx]

    def on_epoch_end(self):
        if self.shuffle:
            self.perm = self.rng.permutation(self.__len__())

    def __getitem__(self, idx: int) -> tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        Returns
        -------
            - tuple, network inputs:
                     - tf.Tensor, inputs batch of shape=(batch_size, maxlen, nb feats)
                     - tf.Tensor, mask batch of shape=(batch_size, maxlen, nb feats)

            - tf.Tensor, targets batch of shape=(batch_size,)
        """
        ii = self.perm[idx]
        batch_x = self.inputs[ii * self.batch_size : (ii + 1) * self.batch_size]
        batch_y = self.targets[ii * self.batch_size : (ii + 1) * self.batch_size]
        # for some reason the output needs explicit casting to tf.Tensors
        return padding(batch_x), float_me(batch_y)

    def __len__(self) -> int:
        """
        Returns the number of batches contained in the generator.

        Returns
        -------
            - int: generator length
        """
        return ceil(len(self.inputs) / self.batch_size)


def get_data(file: Path, geo: Geometry) -> np.ndarray:
    """
    Returns the point cloud from file

    Parameters
    ----------
        - file: Path, the .npz input file path
        - geo: Geometry, object describing detector geometry

    Returns
    -------
        - np.ndarray, of shape=(nb events,) each entry represents a point cloud
                    of shape=([nb hits], nb feats)
    """
    data = scipy.sparse.load_npz(file)
    rows, digits = data.nonzero()
    energies = data.data
    xs_idx = digits // (geo.nb_ybins * geo.nb_zbins)
    mod = digits % (geo.nb_ybins * geo.nb_zbins)
    ys_idx = mod // geo.nb_zbins
    zs_idx = mod % geo.nb_zbins

    xs = geo.xbins[xs_idx] + geo.xbin_w / 2
    ys = geo.ybins[ys_idx] + geo.ybin_w / 2
    zs = geo.zbins[zs_idx] + geo.zbin_w / 2

    pc = np.stack([xs, ys, zs, energies], axis=1)

    splits = np.cumsum(np.bincount(rows))[:-1]
    pc = to_np(np.split(pc, splits))
    return pc


def read_data(folder: Path, setup: dict) -> tuple[Dataset, Dataset, Dataset]:
    """
    Loads data for attention network

    Parameters
    ----------
        - data_folder: Path, the input data folder path
        - setup: dict, settings dictionary

    Returns
    -------
        - Dataset: train generator
        - Dataset: val generator
        - Dataset: test generator
    """
    geo = Geometry(setup["detector"])
    data_sig_l = []
    data_bkg_l = []
    for file in folder.iterdir():
        # signal files
        is_signal = file.name[0] == "b"
        is_background = file.name[0] == "e"
        if is_signal:
            if file.suffix == ".npz":
                data_sig_l.append(get_data(file, geo))
        # background files
        elif is_background:
            if file.suffix == ".npz":
                data_bkg_l.append(get_data(file, geo))

    data_sig_l = np.concatenate(data_sig_l, axis=0)
    data_bkg_l = np.concatenate(data_bkg_l, axis=0)

    logger.debug(f"Signal shapes data: {data_sig_l.shape}")
    logger.debug(f"Background shapes data: {data_bkg_l.shape}")

    data = np.concatenate([data_bkg_l, data_sig_l], axis=0)
    targets = np.concatenate([np.zeros(len(data_bkg_l)), np.ones(len(data_sig_l))])

    logger.debug(f"Data shape: {data.shape}")
    logger.debug(f"Targets shape: {targets.shape}")

    inputs_tv, inputs_test, targets_tv, targets_test = train_test_split(
        data, targets, test_size=0.1, random_state=setup["seed"]
    )

    inputs_train, inputs_val, targets_train, targets_val = train_test_split(
        inputs_tv, targets_tv, test_size=0.1, random_state=setup["seed"]
    )

    batch_size = setup["model"]["attention"]["net_dict"]["batch_size"]

    train_generator = Dataset(
        inputs_train,
        targets_train,
        batch_size,
        True,
        setup["seed"],
    )
    val_generator = Dataset(inputs_val, targets_val, batch_size)
    test_generator = Dataset(inputs_test, targets_test, batch_size)

    return train_generator, val_generator, test_generator
