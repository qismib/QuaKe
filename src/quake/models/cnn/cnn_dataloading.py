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
from quake import PACKAGE
from quake.dataset.generate_utils import Geometry


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


def get_data(file: Path, geo: Geometry) -> np.ndarray:
    """Returns 3D voxelized histograms.

    Parameters
    ----------
    file: Path
        Filename containing sparse data in '.npz' format.
    geo: Geometry
        Object describing detector geometry.

    Returns
    -------
    np.ndarray
        3D voxelized histogram, of shape=(nb events, nb_xbins, nb_ybins, nb_zbins, 1)
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
    # hist3d = matrix.toarray().reshape(shape)
    return [YZ_plane, XZ_plane, XY_plane]


def roll_crop(
    plane: np.ndarray, first_crop_edge: int, second_crop_edge: int
) -> np.ndarray:
    """Shifts tracks to a corner and reduces dimensionality by cropping

    Parameters
    ----------
    plane: nb.ndarray
        2D projection plane, of shape().
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

    data_sig_x, data_sig_y, data_sig_z = [], [], []
    data_bkg_x, data_bkg_y, data_bkg_z = [], [], []

    for file in data_folder.iterdir():
        logger.info("Loading" + str(file))
        is_signal = file.name[0] == "b"
        if file.suffix == ".npz":
            YZ_plane, XZ_plane, XY_plane = get_data(file, geo)
            if is_signal:
                data_sig_x.append(YZ_plane)
                data_sig_y.append(XZ_plane)
                data_sig_z.append(XY_plane)
            else:
                data_bkg_x.append(YZ_plane)
                data_bkg_y.append(XZ_plane)
                data_bkg_z.append(XY_plane)

    data_sig_x = np.concatenate(data_sig_x, axis=0)
    data_sig_y = np.concatenate(data_sig_y, axis=0)
    data_sig_z = np.concatenate(data_sig_z, axis=0)

    data_bkg_x = np.concatenate(data_bkg_x, axis=0)
    data_bkg_y = np.concatenate(data_bkg_y, axis=0)
    data_bkg_z = np.concatenate(data_bkg_z, axis=0)

    projections = [
        np.concatenate([data_sig_x, data_bkg_x]),
        np.concatenate([data_sig_y, data_bkg_y]),
        np.concatenate([data_sig_z, data_bkg_z]),
    ]
    labels = np.concatenate(
        [np.ones(data_sig_x.shape[0]), np.zeros(data_bkg_x.shape[0])]
    )

    if dsetup["should_crop_planes"]:
        # returning the projections cropped around the centre
        # lim_x = (geo.nb_xbins - geo.nb_xbins_reduced)//2
        # lim_y = (geo.nb_ybins - geo.nb_ybins_reduced)//2
        # lim_z = (geo.nb_zbins - geo.nb_zbins_reduced)//2
        # projections[0] = projections[0][:, lim_y:-lim_y, lim_z:-lim_z, :]
        # projections[1] = projections[1][:, lim_x:-lim_x, lim_z:-lim_z, :]
        # projections[2] = projections[2][:, lim_x:-lim_x, lim_y:-lim_y, :]

        projections[0] = roll_crop(
            projections[0], geo.nb_ybins_reduced, geo.nb_zbins_reduced
        )
        projections[1] = roll_crop(
            projections[1], geo.nb_xbins_reduced, geo.nb_zbins_reduced
        )
        projections[2] = roll_crop(
            projections[2], geo.nb_xbins_reduced, geo.nb_ybins_reduced
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
