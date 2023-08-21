"""
    This module implements function for data loading for Autoencoder Network.
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
    get_dataset_balance_message_autoencoder,
    load_splitting_maps,
    save_splitting_maps,
)
from quake import PACKAGE
from quake.dataset.generate_utils import Geometry
from quake.utils.configflow import float_me

logger = logging.getLogger(PACKAGE + ".autoencoder")
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
    """Adds zero-padding to sequences to match the `max_legth` parameter.

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
        [
            np.pad(seq, ((0, max_length - len(seq)), (0, 0)), "constant")
            for seq in inputs
        ]  # "constant"
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
        use_spatial_dims=True,
        spatial_dims=3,
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
        sigmas: tf.Tensor
            The features standard deviations of shape=(nb features).
        use_spatial_dims = bool
            Wether to include hit positions in addition to hit energies.
        seed: int
            Random generator seed for reproducibility.
        """
        self.inputs = to_np(inputs)
        self.classes = targets
        self.batch_size = batch_size
        self.should_standardize = should_standardize
        self.mus = mus
        self.sigmas = sigmas
        self.use_spatial_dims = use_spatial_dims
        self.seed = seed
        self.spatial_dims = spatial_dims

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

        if self.use_spatial_dims:
            self.fixed_length_inputs = self.fixed_length_inputs.reshape(
                self.fixed_length_inputs.shape[0], -1
            )
        else:
            self.fixed_length_inputs = self.fixed_length_inputs[:, :, -1]

        self.inputs = self.fixed_length_inputs
        self.targets = self.fixed_length_inputs
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
    """Loads data for Autoencoder network.

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
        split_ratio = setup["model"]["autoencoder"]["test_split_ratio"]
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

    batch_size = setup["model"]["autoencoder"]["net_dict"]["batch_size"]
    spatial_dims = setup["model"]["autoencoder"]["net_dict"]["spatial_dims"]
    if not (spatial_dims == 2 or spatial_dims == 3):
        logger.error(f"spatial_dims in net_dict should be either 2 or 3")

    if spatial_dims == 2:
        for i, pt_cloud in enumerate(inputs_train):
            inputs_train[i] = merge_rows(pt_cloud[:, [0, 1, 3]])
        for i, pt_cloud in enumerate(inputs_val):
            inputs_val[i] = merge_rows(pt_cloud[:, [0, 1, 3]])
        for i, pt_cloud in enumerate(inputs_test):
            inputs_test[i] = merge_rows(pt_cloud[:, [0, 1, 3]])

    train_generator = Dataset(
        inputs_train,
        targets_train,
        batch_size,
        should_standardize=True,
        use_spatial_dims=setup["model"]["autoencoder"]["net_dict"]["use_spatial_dims"],
        seed=setup["seed"],
    )

    mus = train_generator.mus
    sigmas = train_generator.sigmas

    val_generator = Dataset(
        inputs_val,
        targets_val,
        batch_size,
        should_standardize=True,
        use_spatial_dims=setup["model"]["autoencoder"]["net_dict"]["use_spatial_dims"],
        mus=mus,
        sigmas=sigmas,
    )
    test_generator = Dataset(
        inputs_test,
        targets_test,
        batch_size,
        should_standardize=True,
        use_spatial_dims=setup["model"]["autoencoder"]["net_dict"]["use_spatial_dims"],
        mus=mus,
        sigmas=sigmas,
    )

    logger.info(get_dataset_balance_message_autoencoder(train_generator, "Train"))
    logger.info(get_dataset_balance_message_autoencoder(val_generator, "Validation"))
    logger.info(get_dataset_balance_message_autoencoder(test_generator, "Test"))

    train_max_hits = train_generator.inputs.shape[1]
    val_max_hits = val_generator.inputs.shape[1]
    test_max_hits = test_generator.inputs.shape[1]
    max_input_nb = np.max([train_max_hits, val_max_hits, test_max_hits])

    nhits_pad = (-train_max_hits, -val_max_hits, -test_max_hits) + max_input_nb
    for i in range(nhits_pad[0]):
        train_generator.inputs = np.hstack(
            [train_generator.inputs, np.zeros((train_generator.inputs.shape[0], 1))]
        )  # , train_generator.inputs.shape[2])])
    for i in range(nhits_pad[1]):
        val_generator.inputs = np.hstack(
            [val_generator.inputs, np.zeros((val_generator.inputs.shape[0], 1))]
        )  # , val_generator.inputs.shape[2])])
    for i in range(nhits_pad[2]):
        test_generator.inputs = np.hstack(
            [test_generator.inputs, np.zeros((test_generator.inputs.shape[0], 1))]
        )  # , 1, test_generator.inputs.shape[2])])

    means = train_generator.mus.numpy()
    stds = train_generator.sigmas.numpy()

    train_generator = normalize_dataset(train_generator, means, stds)
    val_generator = normalize_dataset(val_generator, means, stds)
    test_generator = normalize_dataset(test_generator, means, stds)

    # train_generator.inputs = np.hstack([train_generator.nb_hits_array.reshape(-1, 1), train_generator.inputs])
    # val_generator.inputs = np.hstack([val_generator.nb_hits_array.reshape(-1, 1), val_generator.inputs])
    # test_generator.inputs = np.hstack([test_generator.nb_hits_array.reshape(-1, 1), test_generator.inputs])

    train_generator.targets = train_generator.inputs
    val_generator.targets = val_generator.inputs
    test_generator.targets = test_generator.inputs

    val_generator.fixed_length_inputs = val_generator.inputs
    test_generator.fixed_length_inputs = test_generator.inputs
    val_generator.max_hit_length = train_generator.max_hit_length
    test_generator.max_hit_length = train_generator.max_hit_length
    return train_generator, val_generator, test_generator


def normalize_dataset(dataset: Dataset, means: list[np.double], stds: list[np.double]):
    """
    Normalizes x, y, z, e in order to have unitary standard deviation and zero mean.

    Parameters:
    -----------
    dataset: Dataset
        The Dataset object
    means: list(np.double):
        Mean list of training distribution [x_mean, y_mean, z_mean, e_mean]
    stds: list(np.double):
        Std list of training distribution [x_std, y_std, z_std, e_std]

    Returns:
    --------
    dataset: Dataset
        The normalized Dataset object
    """

    dataset.inputs[:, ::4] = (dataset.inputs[:, ::4] - 1.0 * means[0]) / stds[0]
    dataset.inputs[:, 1::4] = (dataset.inputs[:, 1::4] - 1.0 * means[1]) / stds[1]
    dataset.inputs[:, 2::4] = (dataset.inputs[:, 2::4] - 1.0 * means[2]) / stds[2]
    dataset.inputs[:, 3::4] = (dataset.inputs[:, 3::4] - 1.0 * means[3]) / stds[3]

    # nb_features = len(means)
    # for i in range(dataset.inputs.shape[0]):
    #     nonpadded_idx = np.arange(dataset.nb_hits_array[i] * nb_features)
    #     for j in range(dataset.nb_features):
    #         max_entry, min_entry = (
    #             dataset.inputs[i, nonpadded_idx[j::nb_features]].max(),
    #             dataset.inputs[i, nonpadded_idx[j::nb_features]].min(),
    #         )

    #         if max_entry - min_entry != 0:
    #             dataset.inputs[i, nonpadded_idx[j::nb_features]] = (
    #                 dataset.inputs[i, nonpadded_idx[j::nb_features]] - min_entry
    #             ) / (max_entry - min_entry)
    # import pdb; pdb.set_trace()
    # dataset.inputs[:, nb_features - 1 :: nb_features] = (
    #     dataset.inputs[:, nb_features - 1 :: nb_features]
    #     / dataset.inputs[:, nb_features - 1 :: nb_features].max(axis=1)[:, None]
    # )
    return dataset


def merge_rows(matrix: np.ndarray) -> np.ndarray:
    """
    Merge rows in a matrix based on the first and second column values.

    If the first and second column values are equal in two or more consecutive rows, they are merged into one row
    where the first and second column values remain the same, and the third column value is the sum of the
    entries of the previous rows.

    Parameters:
    matrix: numpy.ndarray
        The input matrix to be processed.

    Returns:
    numpy.ndarray
        The merged matrix after combining rows with equal first and second column values.
    """
    merged_matrix = []
    current_row = matrix[0].copy()
    for row in matrix[1:]:
        if (row[0] == current_row[0]) and (row[1] == current_row[1]):
            current_row[2] += row[2]
        else:
            merged_matrix.append(current_row)
            current_row = row.copy()
    merged_matrix.append(current_row)
    return np.array(merged_matrix)


def merge_all_matrices(arr: np.array) -> np.ndarray:
    """
    Merge rows in all matrices within the input ndarray based on the first and second column values.

    This function processes each matrix in the input ndarray separately using the `merge_rows` function
    to merge rows with equal first and second column values.

    Parameters:
    -----------
    arr: numpy.ndarray
        The input ndarray containing nb_hitsx3 matrices.

    Returns:
    ---------
    numpy.ndarray
        The ndarray containing the merged matrices after combining rows with equal first and
        second column values.
    """
    merged_matrices = []
    for matrix in arr:
        merged_matrix = merge_rows(matrix)
        merged_matrices.append(merged_matrix)
    return np.array(merged_matrices)
