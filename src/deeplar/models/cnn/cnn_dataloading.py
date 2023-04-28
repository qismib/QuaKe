"""
    This module implements function for data loading for CNN.
"""
import logging
from typing import Tuple
from pathlib import Path
from math import ceil
import numpy as np
import tensorflow as tf
import scipy
from ..utils import (
    dataset_split_util,
    get_dataset_balance_message,
    load_splitting_maps,
    save_splitting_maps,
)
from deeplar import PACKAGE
from deeplar.dataset.generate_utils import Geometry


logger = logging.getLogger(PACKAGE + ".cnn")


class Dataset(tf.keras.utils.Sequence):
    """Dataset sequence for CNN Network."""

    def __init__(
        self,
        inputs: list[np.ndarray],
        targets: np.ndarray,
        batch_size: int,
        should_standardize: bool = True,
        mus: tf.Tensor = None,
        sigmas: tf.Tensor = None,
        shuffle: bool = False,
        seed: int = 12345,
    ):
        """
        Parameters
        ----------
        inputs: list
            Arrays of input projections, each of shape=(nb events,H,W,1).
        targets: np.ndarray
            Array of labels, of shape=(nb events,).
        batch_size: int
            The batch size.
        should_standardize:
            Wether to standardize the inputs or not.
        mus: list[int]
            The pixels means for the three projections [yz, xz, xy].
        stds: list[int]
            The pixels standard deviations for the three projections [yz, xz, xy].
        shuffle: bool
            Wether to shuffle dataset at epoch end or not.
        seed:
            Random generator seed for reproducibility.
        """
        # self.inputs = to_np(inputs)
        self.projections = inputs
        self.yz_planes, self.xz_planes, self.xy_planes = self.projections
        self.targets = targets
        self.batch_size = batch_size
        self.should_standardize = should_standardize
        self.mus = mus
        self.sigmas = sigmas
        self.shuffle = shuffle
        self.seed = seed

        if self.should_standardize:
            if self.mus is None or self.sigmas is None:
                self.mus = list(map(np.mean, self.projections))
                self.sigmas = list(map(np.std, self.projections))

        self.data_len = len(self.targets)
        self.indices = np.arange(self.data_len)
        self.rng = np.random.default_rng(self.seed)

    def on_epoch_end(self):
        """Random shuffles dataset examples on epoch end.

        Behavior depends on self.shuffle boolean flag.
        """
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """Returns a batch of inputs and labels.

        Parameters
        ----------
        idx: int
            The batch number drawn by the generator.

        Returns
        -------
        network inputs: Tuple[np.ndarray, np.ndarray, np.ndarray]
            Three batched 2D projections, of shape=(B,H,W,C).
        targets: np.ndarray
            Batch of labelse, of shape=(batch_size,).
        """
        # define the slice
        s = np.s_[idx * self.batch_size : (idx + 1) * self.batch_size]
        idxs = self.indices[s]
        batch_x = (self.yz_planes[idxs], self.xz_planes[idxs], self.xy_planes[idxs])
        batch_y = self.targets[idxs]
        return batch_x, batch_y

    def __len__(self) -> int:
        """Returns the number of batches contained in the generator.

        Returns
        -------
            - generator length
        """
        return ceil(self.data_len / self.batch_size)

    def get_extra_features(self) -> np.ndarray:
        """Computes custom extra features from events.

        Extra features are:

        - number of active pixels in yz 2D projection
        - number of active pixels in xz 2D projection
        - number of active pixels in xy 2D projection
        - total energy in the event

        Returns
        -------
        extra_features: np.ndarray
            Number of active pixels and tot energy for each event, of
            shape=(nb events, 4).
        """
        nb_active_yz = np.count_nonzero(self.yz_planes, axis=(1, 2, 3))
        nb_active_xz = np.count_nonzero(self.xz_planes, axis=(1, 2, 3))
        nb_active_xy = np.count_nonzero(self.xy_planes, axis=(1, 2, 3))
        tot_energy = self.yz_planes.sum((1, 2, 3))
        extra_features = np.stack(
            [nb_active_xy, nb_active_xz, nb_active_yz, tot_energy], axis=1
        )
        return extra_features


def get_data(file: Path, geo: Geometry, dsetup: dict) -> np.ndarray:
    """Returns 2D projected histograms.

    Parameters
    ----------
    file: Path
        Filename containing sparse data in '.npz' format.
    geo: Geometry
        Object describing detector geometry.
    dsetup: dict
        Detector settings dictionary.
    Returns
    -------
    np.ndarray
        2D projections.
    """
    matrix = scipy.sparse.load_npz(file)
    # shape = (-1, geo.nb_xbins, geo.nb_ybins, geo.nb_zbins, 1)
    shape = (-1, geo.nb_xbins, geo.nb_ybins, geo.nb_zbins)
    # matrix = matrix.tocoo()
    indices = np.mat([matrix.row, matrix.col]).transpose()
    matrix = tf.SparseTensor(indices, matrix.data, matrix.shape)
    matrix = tf.sparse.reshape(matrix, (shape))
    YZ_plane = np.expand_dims(tf.sparse.reduce_sum(matrix, axis=1), 3)
    XZ_plane = np.expand_dims(tf.sparse.reduce_sum(matrix, axis=2), 3)
    XY_plane = np.expand_dims(tf.sparse.reduce_sum(matrix, axis=3), 3)

    if dsetup["should_crop_planes"]:
        YZ_plane = roll_crop(
            np.copy(YZ_plane), geo.nb_ybins_reduced, geo.nb_zbins_reduced
        )
        XZ_plane = roll_crop(
            np.copy(XZ_plane), geo.nb_xbins_reduced, geo.nb_zbins_reduced
        )
        XY_plane = roll_crop(
            np.copy(XY_plane), geo.nb_xbins_reduced, geo.nb_ybins_reduced
        )
    # hist3d = matrix.toarray().reshape(shape)
    return [YZ_plane, XZ_plane, XY_plane]


def get_n_evts(data_folder: Path, geo: Geometry) -> int:
    n_evts_sig = 0
    n_evts_bkg = 0
    for file in data_folder.iterdir():
        n_evts_file = scipy.sparse.load_npz(file).shape[0]
        is_signal = file.name[0] == "b"
        if file.suffix == ".npz":
            if is_signal:
                n_evts_sig = n_evts_sig + n_evts_file
            else:
                n_evts_bkg = n_evts_bkg + n_evts_file
    return n_evts_sig, n_evts_bkg


def roll_crop(
    plane: np.ndarray, first_crop_edge: int, second_crop_edge: int
) -> np.ndarray:
    """Shifts tracks to a corner and reduces dimensionality by cropping

    Parameters
    ----------
    plane: nb.ndarray
        2D projection plane.
    first_crop_edge: int
        Numbers of bins to keep on the first axis.
    second_crop_edge: int
        Numbers of bins to keep on the second axis.

    Returns
    -------
    np.ndarray:
        The cropped planes, of shape=(nb events, first_crop_edge, second_crop_edge, C)
    """
    for i, particle in enumerate(plane):
        c = np.argwhere(particle)
        if list(c):
            cm1 = np.min(c[:, 0])
            cm2 = np.min(c[:, 1])
            plane[i] = np.roll(plane[i], -cm1, axis=0)
            plane[i] = np.roll(plane[i], -cm2, axis=1)
    plane = plane[:, :first_crop_edge, :second_crop_edge]
    return plane


def load_projections_and_labels(
    data_folder: Path,
    dsetup: dict,
) -> Tuple[list[np.ndarray], list[np.ndarray]]:
    """Returns 2D projections from the voxelized data.

    Parameters
    ----------
    folder: Path
        The input data folder path.
    dsetup: dict
        Detector settings dictionary.
    msetup: dict
        CNN model settings dictionary.

    Returns
    -------
    projections: list[np.ndarray]
        [x, y, z] 2D projections of voxelized signal histograms
    labels: list[np.ndarray]
        [x, y, z] 2D projections of voxelized background histograms
    """
    logger.debug("Loading data ...")
    geo = Geometry(dsetup)

    n_evts_sig, n_evts_bkg = get_n_evts(data_folder, geo)
    data_sig_x, data_sig_y, data_sig_z = (
        np.zeros((n_evts_sig, geo.nb_ybins_reduced, geo.nb_zbins_reduced, 1)),
        np.zeros((n_evts_sig, geo.nb_xbins_reduced, geo.nb_zbins_reduced, 1)),
        np.zeros((n_evts_sig, geo.nb_xbins_reduced, geo.nb_ybins_reduced, 1)),
    )
    data_bkg_x, data_bkg_y, data_bkg_z = (
        np.zeros((n_evts_bkg, geo.nb_ybins_reduced, geo.nb_zbins_reduced, 1)),
        np.zeros((n_evts_bkg, geo.nb_xbins_reduced, geo.nb_zbins_reduced, 1)),
        np.zeros((n_evts_bkg, geo.nb_xbins_reduced, geo.nb_ybins_reduced, 1)),
    )

    counter_sig = 0
    counter_bkg = 0
    for file in data_folder.iterdir():
        logger.info(f"Loading {file}")
        is_signal = file.name[0] == "b"
        if file.suffix == ".npz":
            YZ_plane, XZ_plane, XY_plane = get_data(file, geo, dsetup)
            if is_signal:
                data_sig_x[counter_sig : counter_sig + YZ_plane.shape[0]] = YZ_plane
                data_sig_y[counter_sig : counter_sig + XZ_plane.shape[0]] = XZ_plane
                data_sig_z[counter_sig : counter_sig + XY_plane.shape[0]] = XY_plane
                counter_sig = counter_sig + YZ_plane.shape[0]
            else:
                data_bkg_x[counter_bkg : counter_bkg + YZ_plane.shape[0]] = YZ_plane
                data_bkg_y[counter_bkg : counter_bkg + XZ_plane.shape[0]] = XZ_plane
                data_bkg_z[counter_bkg : counter_bkg + XY_plane.shape[0]] = XY_plane
                counter_bkg = counter_bkg + YZ_plane.shape[0]

    projections = [
        np.concatenate([data_sig_x, data_bkg_x]),
        np.concatenate([data_sig_y, data_bkg_y]),
        np.concatenate([data_sig_z, data_bkg_z]),
    ]
    labels = np.concatenate(
        [np.ones(data_sig_x.shape[0]), np.zeros(data_bkg_x.shape[0])]
    )

    return projections, labels


def read_data(
    data_folder: Path, train_folder: Path, setup: dict, split_from_maps: bool = False
) -> Tuple[Dataset, Dataset, Dataset]:
    """Loads data for CNN Network.

    Returns the `tf.keras.utils.Sequence` generators for train, validation and
    train dataset.

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
    (yz_planes, xz_planes, xy_planes), labels = load_projections_and_labels(
        data_folder,
        setup["detector"],
    )

    if split_from_maps:
        logger.info(f"Loading splitting maps from folder: {train_folder}")
        train_map, val_map, test_map = load_splitting_maps(train_folder)
        train_labels = labels[train_map]
        val_labels = labels[val_map]
        test_labels = labels[test_map]
    else:
        yz_train_wrap, yz_val_wrap, yz_test_wrap = dataset_split_util(
            yz_planes,
            labels,
            split_ratio=setup["model"]["attention"]["test_split_ratio"],
            seed=setup["seed"],
            with_indices=True,
        )

        _, train_labels, train_map = yz_train_wrap
        _, val_labels, val_map = yz_val_wrap
        _, test_labels, test_map = yz_test_wrap

        save_splitting_maps(train_folder, train_map, val_map, test_map)
        logger.info(f"Saving splitting maps in folder {train_folder}")

    inputs_train = [yz_planes[train_map], xz_planes[train_map], xy_planes[train_map]]
    train_generator = Dataset(
        inputs_train,
        train_labels,
        setup["model"]["cnn"]["net_dict"]["batch_size"],
        should_standardize=True,
        shuffle=True,
        seed=setup["seed"],
    )

    mus = train_generator.mus
    sigmas = train_generator.sigmas

    inputs_val = [yz_planes[val_map], xz_planes[val_map], xy_planes[val_map]]
    val_generator = Dataset(
        inputs_val,
        val_labels,
        setup["model"]["cnn"]["net_dict"]["batch_size"],
        should_standardize=True,
        mus=mus,
        sigmas=sigmas,
    )
    inputs_test = [yz_planes[test_map], xz_planes[test_map], xy_planes[test_map]]
    test_generator = Dataset(
        inputs_test,
        test_labels,
        setup["model"]["cnn"]["net_dict"]["batch_size"],
        should_standardize=True,
        mus=mus,
        sigmas=sigmas,
    )

    logger.info(get_dataset_balance_message(train_generator, "Train"))
    logger.info(get_dataset_balance_message(val_generator, "Validation"))
    logger.info(get_dataset_balance_message(test_generator, "Test"))

    return train_generator, val_generator, test_generator
