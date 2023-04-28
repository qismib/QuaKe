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
import tensorflow as tf
from ..utils import (
    dataset_split_util,
    get_dataset_balance_message,
    load_splitting_maps,
    save_splitting_maps,
)
from deeplar import PACKAGE
from deeplar.dataset.generate_utils import Geometry
from deeplar.utils.configflow import float_me

logger = logging.getLogger(PACKAGE + ".attention")
to_np = lambda x: np.array(x, dtype=object)


def restore_order(array: np.ndarray, ordering: np.ndarray) -> np.ndarray:
    """In place back projection to input original order before dataset sorting.

    This is useful when predicting network results and initial order matters.

    Parameters
    ----------
    array: np.ndarray
        Iterable to be sorted back of shape=(nb events).
    ordering: np.ndarray
        The array that originally sorted the data of shape=(nb events).

    Returns
    -------
    np.ndarray
        the originally ordered array.
    """
    return np.put_along_axis(array, ordering, array, axis=0)


def fix_sequence_lengths(inputs: np.ndarray, max_length: int = None):
    """Repeats sequences last items to match the `max_legth` parameter.

    This function is used to get a rectangular array from a jagged one.

    Parameters
    ----------
    inputs: np.ndarray
        The inputs sequences, of shape=(nb seq, [seq length], nb features).
    max_length: int
        The sequences maximum length. If `None`, it is computed inside the
        function.

    Returns
    -------
    np.ndarray
        Fixed length sequences, of shape=(nb seq, max_length, nb features).
    """
    if max_length is None:
        seq_lengths = np.array([seq.shape[0] for seq in inputs])
        max_length = np.max(seq_lengths)

    fixed_length = np.stack(
        [np.pad(seq, ((0, max_length - len(seq)), (0, 0)), "edge") for seq in inputs]
    )
    return fixed_length


def standardize_batch(batch: tf.Tensor, mus: tf.Tensor, sigmas: tf.Tensor) -> tf.Tensor:
    """Standardize input features to have zero mean and unit standard deviation.

    Parameters
    ----------
    inputs: tf.Tensor
        Inputs batch of shape=(events), each of shape=([nb hits], nb feats).
    mus: tf.Tensor
        The feature means of shape=(nb_feats,).
    sigmas: tf.Tensor
        The feature standard deviations of shape=(nb_feats,).

    Returns
    -------
    z_scores: tf.Tensor
        The normalized batch of shape=(batch_size, maxlen, nb feats).
    """
    mus = tf.expand_dims(mus, axis=0)
    sigmas = tf.expand_dims(sigmas, axis=0)
    z_scores = (batch - mus) / sigmas
    return z_scores


class Dataset(tf.keras.utils.Sequence):
    """Dataset sequence."""

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
        should_standardize: bool = True,
        mus: tf.Tensor = None,
        sigmas: tf.Tensor = None,
        seed: int = 12345,
    ):
        """
        Parameters
        ----------
        inputs: np.ndarray
            Array of objects, each of shape=([nb hits], nb features).
        targets: np.ndarray
            Array of shape=(nb events).
        batch_size: int
            The batch size.
        should_standardize: bool
            Wether to standardize the inputs or not.
        mus: tf.Tensor
            The features means of shape=(nb features).
        stds: tf.Tensor
            The features standard deviations of shape=(nb features).
        seed: int
            Random generator seed for reproducibility.
        """
        self.inputs = to_np(inputs)
        self.targets = targets
        self.batch_size = batch_size
        self.should_standardize = should_standardize
        self.mus = mus
        self.sigmas = sigmas
        self.seed = seed

        self.nb_hits_array = np.array([ev.shape[0] for ev in self.inputs])
        self.max_hit_length = np.max(self.nb_hits_array)

        self.nb_features = self.inputs[0].shape[-1]

        if self.should_standardize:
            if self.mus is None or self.sigmas is None:
                data = np.concatenate(self.inputs).reshape([-1, self.nb_features])
                self.mus = float_me(data.mean(0))
                self.sigmas = float_me(data.std(0))

        self.fixed_length_inputs = fix_sequence_lengths(
            self.inputs, self.max_hit_length
        )

        self.data_len = len(self.targets)
        self.indices = np.arange(self.data_len)
        self.rng = np.random.default_rng(self.seed)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns a batch of inputs and labels.
        Parameters
        ----------
        idx: int
            The batch number drawn by the generator.
        Returns
        -------
        network inputs: Tuple[np.ndarray]
            Three batched 3D point cloud, of shape=(batch_size ,max_length, nb features).
        targets: np.ndarray
            Batch of labelse, of shape=(batch_size,).
        """
        # define the slice
        s = np.s_[idx * self.batch_size : (idx + 1) * self.batch_size]
        idxs = self.indices[s]
        batch_x = self.fixed_length_inputs[idxs]
        batch_y = self.targets[idxs]
        return batch_x, batch_y

    def __len__(self) -> int:
        """Returns the number of batches contained in the generator.

        Returns
        -------
        int:
            Generator length.
        """
        return ceil(len(self.inputs) / self.batch_size)

    def get_extra_features(self) -> np.ndarray:
        """Computes custom extra features from events.

        Extra features are:

        - number of active pixels in 3D event
        - total energy in the event

        Returns
        -------
        extra_features: np.ndarray
            Number of active pixels and tot energy for each event, of
            shape=(nb events, 2).
        """
        # TODO: maybe use awkward arrays to do this
        tot_energy = [event[:, -1].sum() for event in self.inputs]
        extra_features = np.stack([self.nb_hits_array, tot_energy], axis=1)
        return extra_features


def get_data(file: Path, geo: Geometry) -> np.ndarray:
    """Returns the point cloud from file.

    Parameters
    ----------
    file: Path
        The .npz input file path.
    geo: Geometry
        Object describing detector geometry.

    Returns
    -------
    point cloud: np.ndarray
        array of objects of shape=(nb events,) each entry represents a point
        cloud of shape=([nb hits], nb feats).
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


def read_data(
    data_folder: Path, train_folder: Path, setup: dict, split_from_maps: bool = False
) -> Tuple[Dataset, Dataset, Dataset]:
    """Loads data for attention network.

    Parameters
    ----------
    data_folder: Path
        The input data folder path.
    train_folder: Path
        The train output folder path.
    setup: dict
        Settings dictionary.
    split_from_maps: bool
        Wether to load splitting maps from file to restore train/val/test
        datasets from a previous splitting.

    Returns
    -------
    train_generator: Dataset
        Train generator.
    val_generator: Dataset
        Validation generator.
    test_generator: Dataset
        Test generator.
    """
    geo = Geometry(setup["detector"])
    data_sig_l = []
    data_bkg_l = []
    for file in data_folder.iterdir():
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

    data = np.concatenate([data_sig_l, data_bkg_l], axis=0)
    targets = np.concatenate([np.ones(len(data_sig_l)), np.zeros(len(data_bkg_l))])

    logger.debug(f"Data shape: {data.shape}")
    logger.debug(f"Targets shape: {targets.shape}")

    if split_from_maps:
        logger.info(f"Loading splitting maps from folder: {train_folder}")
        train_map, val_map, test_map = load_splitting_maps(train_folder)
        inputs_train = data[train_map]
        inputs_val = data[val_map]
        inputs_test = data[test_map]
        targets_train = targets[train_map]
        targets_val = targets[val_map]
        targets_test = targets[test_map]
    else:
        split_ratio = setup["model"]["attention"]["test_split_ratio"]
        train_wrap, val_wrap, test_wrap = dataset_split_util(
            data,
            targets,
            split_ratio=split_ratio,
            seed=setup["seed"],
            with_indices=True,
        )
        inputs_train, targets_train, train_map = train_wrap
        inputs_val, targets_val, val_map = val_wrap
        inputs_test, targets_test, test_map = test_wrap

        save_splitting_maps(train_folder, train_map, val_map, test_map)
        logger.info(f"Saving splitting maps in folder {train_folder}")

    batch_size = setup["model"]["attention"]["net_dict"]["batch_size"]

    train_generator = Dataset(
        inputs_train,
        targets_train,
        batch_size,
        should_standardize=True,
        seed=setup["seed"],
    )

    mus = train_generator.mus
    sigmas = train_generator.sigmas

    val_generator = Dataset(
        inputs_val,
        targets_val,
        batch_size,
        should_standardize=True,
        mus=mus,
        sigmas=sigmas,
    )
    test_generator = Dataset(
        inputs_test,
        targets_test,
        batch_size,
        should_standardize=True,
        mus=mus,
        sigmas=sigmas,
    )

    logger.info(get_dataset_balance_message(train_generator, "Train"))
    logger.info(get_dataset_balance_message(val_generator, "Validation"))
    logger.info(get_dataset_balance_message(test_generator, "Test"))

    return train_generator, val_generator, test_generator
