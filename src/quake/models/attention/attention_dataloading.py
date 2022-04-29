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


def standardize_batch(batch: tf.Tensor, mus: tf.Tensor, sigmas: tf.Tensor) -> tf.Tensor:
    """
    Standardize input features to have zero mean and unit standard deviation.

    Parameters
    ----------
        - inptus: inputs batch of shape=(events), each of shape=([nb hits], nb feats)
        - mus: the feature means of shape=(nb_feats,)
        - sigmas: the feature standard deviations of shape=(nb_feats,)
    Returns
    -------
        - the normalized batch of shape=(batch_size, maxlen, nb feats)
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
        smart_batching: bool = False,
        should_standardize: bool = True,
        mus: tf.Tensor = None,
        sigmas: tf.Tensor = None,
        seed: int = 12345,
    ):
        """
        Parameters
        ----------
            - inputs: array of objects, each of shape=([nb hits], nb features)
            - targets: array of shape=(nb events)
            - batch_size: the batch size
            - smart_batching: wether to sample with smart batch algorithm
            - should_standardize: wether to standardize the inputs or not
            - mus: the features means of shape=(nb features)
            - stds: the features standard deviations of shape=(nb features)
            - seed: random generator seed for reproducibility
        """
        self.inputs = to_np(inputs)
        self.targets = targets
        self.batch_size = batch_size
        self.smart_batching = smart_batching
        self.should_standardize = should_standardize
        self.mus = mus
        self.sigmas = sigmas
        self.seed = seed

        self.nb_features = self.inputs[0].shape[-1]

        if self.should_standardize:
            if self.mus is None or self.sigmas is None:
                data = np.concatenate(self.inputs).reshape([-1, self.nb_features])
                self.mus = float_me(data.mean(0))
                self.sigmas = float_me(data.std(0))

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

        Note
        ----

        The input feature axis contains the following data:
            - normalized x coordinate [mm]
            - normalized y coordinate [mm]
            - normalized z coordinate [mm]
            - normalized pixel energy value [MeV]
        """
        # smart batching
        if self.smart_batching:
            ii = self.sample_smart_batch(idx)
        else:
            ii = np.s_[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = self.inputs[ii]
        batch_y = self.targets[ii]
        # for some reason the output requires explicit casting to tf.Tensors
        padded_batch, masks = padding(batch_x)
        float_target = float_me(batch_y)

        norm_batch = (
            standardize_batch(padded_batch, self.mus, self.sigmas)
            if self.should_standardize
            else padded_batch
        )

        return (norm_batch, masks), float_target

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

    data = np.concatenate([data_bkg_l, data_sig_l], axis=0)
    targets = np.concatenate([np.zeros(len(data_bkg_l)), np.ones(len(data_sig_l))])

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
        smart_batching=True,
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
