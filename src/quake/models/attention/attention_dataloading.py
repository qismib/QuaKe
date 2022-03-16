"""
    This module implements function for data loading for AttentionNetwork.
    Dynamic batching is employed.
"""
import logging
from typing import Tuple
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


def restore_order(array: np.ndarray, ordering: np.ndarray) -> np.ndarray:
    """
    In place back projection to input original order before dataset sorting.
    This is useful when predicting network results and initial order matters.

    Parameters
    ----------
        - array: iterable to be sorted back of shape=(nb events)
        - ordering: the array that originally sorted the data of shape=(nb events)

    Returns
    -------
        - the originally ordered array
    """
    return np.put_along_axis(array, ordering, array, axis=0)


def padding(array: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Pads the inputs to fit data into a tensor.

    Parameters
    ----------
        - array: iterable of objects each of shape=([nb hits], nb features)

    Returns
    -------
        - padded data of shape=(maxlen, nb features)
        - mask of shape=(maxlen, maxlen)
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
        smart_batching: bool = False,
        seed: int = 12345,
    ):
        """
        Parameters
        ----------
            - inputs: array of objects, each of shape=([nb hits], nb features)
            - targets: array of shape=(nb events)
            - batch_size: the batch size
            - smart_batching: wether to sample with smart batch algorithm
            - seed: random generator seed for reproducibility

        """
        self.inputs = to_np(inputs)
        self.targets = targets
        self.batch_size = batch_size
        self.smart_batching = smart_batching
        self.seed = seed

        self.data_len = len(self.targets)
        self.available_idxs = np.arange(self.data_len)
        self.rng = np.random.default_rng(self.seed) if self.smart_batching else None
        self.sort_data()

        # Model.fit calls samples a batch first, spoiling the remaining batches
        self.is_first_pass = True

    def sort_data(self):
        """
        Sorts inputs and targets according to increasing number of hits in
        event. This is needed for dynamic batching.
        """
        fn = lambda pair: len(pair[0])
        indices = np.arange(self.data_len)
        self.sorting_idx = [i for _, i in sorted(zip(self.inputs, indices), key=fn)]
        self.inputs = self.inputs[self.sorting_idx]
        self.targets = self.targets[self.sorting_idx]

    def on_epoch_end(self):
        if self.smart_batching:
            assert (
                len(self.available_idxs) == 0
            ), f"Remaining points is not zero, found {len(self.available_idxs)}"
            self.available_idxs = np.arange(self.data_len)

    def __getitem__(self, idx: int) -> Tuple[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        Parameters
        ----------
            - idx: the number of batch drawn by the generator

        Returns
        -------
            - network inputs:
                - inputs batch of shape=(batch_size, maxlen, nb feats)
                - mask batch of shape=(batch_size, maxlen, nb feats)

            - targets batch of shape=(batch_size,)
        """
        # smart batching
        if self.smart_batching:
            ii = self.sample_smart_batch(idx)
        else:
            ii = np.s_[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_x = self.inputs[ii]
        batch_y = self.targets[ii]

        # for some reason the output requires explicit casting to tf.Tensors
        return padding(batch_x), float_me(batch_y)

    def sample_smart_batch(self, idx: int) -> np.ndarray:
        """
        Returns indices of the smart batch samples. Smart batching randomly
        draws an example `i` from the available training points, then samples
        the batch with the [i: i + batch_size] slice.

        Parameters
        ----------
            - idx: the number of batch drawn by the generator

        Returns
        -------
            - the array of indices to sample the smart batch of shape=(batch size)
        """
        nb_available = len(self.available_idxs)
        if nb_available <= self.batch_size:
            # last remaining batch
            assert (
                idx == len(self) - 1
            ), f"Batch index {idx + 1}/{len(self)}, remaining {nb_available}, batch size {self.batch_size}"
            sampled = np.s_[:]
            ii = self.available_idxs
        else:
            i = self.rng.integers(0, nb_available - self.batch_size)
            sampled = np.s_[i : i + self.batch_size]
            ii = self.available_idxs[sampled]
            # skip the first pass deletion
        if self.is_first_pass:
            self.is_first_pass = False
        else:
            self.available_idxs = np.delete(self.available_idxs, sampled)
        return ii

    def __len__(self) -> int:
        """
        Returns the number of batches contained in the generator.

        Returns
        -------
            - generator length
        """
        return ceil(len(self.inputs) / self.batch_size)


def get_data(file: Path, geo: Geometry) -> np.ndarray:
    """
    Returns the point cloud from file

    Parameters
    ----------
        - file: the .npz input file path
        - geo: object describing detector geometry

    Returns
    -------
        - array of objects of shape=(nb events,) each entry represents a point
          cloud of shape=([nb hits], nb feats)
    """
    data = scipy.sparse.load_npz(file)
    evt, coords = data.nonzero()
    energies = np.array(data.tocsr()[evt, coords])[0]

    xs_idx, mod = divmod(coords, geo.nb_ybins * geo.nb_zbins)
    ys_idx, zs_idx = divmod(mod, geo.nb_zbins)

    xs = geo.xbins[xs_idx] + geo.xbin_w / 2
    ys = geo.ybins[ys_idx] + geo.ybin_w / 2
    zs = geo.zbins[zs_idx] + geo.zbin_w / 2

    pc = np.stack([xs, ys, zs, energies], axis=1)

    splits = np.cumsum(np.bincount(evt))[:-1]
    pc = to_np(np.split(pc, splits))

    return pc


def read_data(folder: Path, setup: dict) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Loads data for attention network

    Parameters
    ----------
        - data_folder: the input data folder path
        - setup: settings dictionary

    Returns
    -------
        - train generator
        - val generator
        - test generator
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
        smart_batching=True,
        seed=setup["seed"],
    )
    val_generator = Dataset(inputs_val, targets_val, batch_size)
    test_generator = Dataset(inputs_test, targets_test, batch_size)

    print_dataset_balance(train_generator, "Train")
    print_dataset_balance(val_generator, "Validation")
    print_dataset_balance(test_generator, "Test")

    return train_generator, val_generator, test_generator


def print_dataset_balance(dataset: Dataset, name: str):
    """
    Logs the dataset balancing between classes

    Parameters
    ----------
        - dataset: the dataset to log
        - name: the dataset name to be logged
    """
    nb_examples = dataset.data_len
    positives = np.count_nonzero(dataset.targets)
    logger.info(
        f"{name} dataset balancing: {nb_examples} training points, "
        f"of which {positives/nb_examples*100:.2f}% positives"
    )
