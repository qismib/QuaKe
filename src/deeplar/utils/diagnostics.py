""" This module contains utility function to make diagnostics plots. """
import io
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from deeplar.utils.configflow import bool_me
from .configflow import arrayLike


def histogram_activations_image(y_pred: arrayLike, y_true: arrayLike) -> plt.Figure:
    """
    Returns a histogram of the network classification scores.
    Overlapped histograms scores for the positive and negative samples.
    Parameters
    ----------
    y_pred: arrayLike
        The tensor of predicted scores shape=(nb points,).
    y_true: arrayLike
        The tensor of ground truths of shape=(nb points,).
    Returns
    -------
    figure: plt.Figure
        The pyplot histograms.
    """
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()

    y_true = bool_me(y_true).numpy()

    scores_true = y_pred[y_true]
    scores_false = y_pred[~y_true]

    tp = np.count_nonzero(scores_true > 0.5)
    fp = np.count_nonzero(scores_false < 0.5)
    tot = len(y_pred)
    accuracy = (tp + fp) / tot

    bins = np.linspace(0, 1, 50)
    h_true, _ = np.histogram(scores_true, bins=bins)
    h_false, _ = np.histogram(scores_false, bins=bins)
    figure = plt.figure(figsize=(8, 4))
    ax = figure.add_subplot()
    ax.title.set_text(f"Network classification. Accuracy: {accuracy*100:.3f}%")
    ax.hist(
        bins[:-1],
        bins,
        weights=h_true,
        histtype="bar",
        alpha=0.6,
        lw=0.5,
        color="#ff7f0e",
        label=r"$0\nu\beta\beta$",
        density=True,
        linewidth=1.0,
        edgecolor="black",
    )
    ax.hist(
        bins[:-1],
        bins,
        weights=h_false,
        histtype="bar",
        alpha=0.6,
        lw=0.5,
        color="#1f77b4",
        label=r"$e^-$",
        density=True,
        linewidth=1.0,
        edgecolor="black",
    )
    ax.set_xlabel("Network score")
    bw = bins[-1] - bins[-2]
    ax.set_ylabel(f"Event ratio [%] per {bw:.2f} score interval")
    ax.legend()
    return figure


def scatterplot_features_image(features: arrayLike, y_true: arrayLike) -> plt.Figure:
    """Returns a 2D scatterplot of the input features.
    Useful to look visually the separation in the 2D plane for binary
    classification.
    The input features should have shape `(nb points, nb dims)`.
    If `nb dims == 2`, plot features in 2D plane.
    If `nb dims == 3`, plot features in 3D space.
    If `nb dims > 3`, use tSNE technique to reduce point to 2 dimensions and
    plot into 2D plane.
    Parameters
    ----------
    features: arrayLike
        The tensor of features shape=(nb points, nb_features).
    y_true: arrayLike
        The tensor of ground truths of shape=(nb points,).
    Returns
    -------
    figure: plt.Figure
        The pyplot scatterplot.
    """
    y_true = bool_me(y_true).numpy()

    if isinstance(features, tf.Tensor):
        features = features.numpy()
    mean = features.mean(0, keepdims=True)
    std = features.std(0, keepdims=True)
    features = (features - mean) / std
    nb_dims = features.shape[1]

    y_true = bool_me(y_true).numpy()

    projection = "3d" if nb_dims == 3 else None  # use 3D space only if nb_dims is 3
    # define scatter function arguments depending on point dimensionality
    if 1 < nb_dims <= 3:
        pos = features[y_true]
        neg = features[~y_true]
        if nb_dims == 2:
            pos_args = (pos[:, 0], pos[:, 1])
            neg_args = (neg[:, 0], neg[:, 1])
        if nb_dims == 3:
            pos_args = (pos[:, 0], pos[:, 1], pos[:, 2])
            neg_args = (neg[:, 0], neg[:, 1], neg[:, 2])
    elif nb_dims > 3:
        transformed = TSNE(
            n_components=2, learning_rate="auto", init="random"
        ).fit_transform(features)
        pos = transformed[y_true]
        neg = transformed[~y_true]
        pos_args = (pos[:, 0], pos[:, 1])
        neg_args = (neg[:, 0], neg[:, 1])
    else:
        raise ValueError(f"The extracted features must be more than 1, got {nb_dims}")

    figure = plt.figure()
    ax = figure.add_subplot(projection=projection)
    ax.title.set_text("Standardized network extracted features")
    ax.scatter(*pos_args, c="green", s=10, label=r"$0\nu\beta\beta$")
    ax.scatter(*neg_args, c="red", s=10, label=r"$e^-$")
    ax.set_xlim([-3.5, 3.5])
    ax.set_ylim([-3.5, 3.5])
    ax.set_xlabel(r"$1^{st}$ feature $[\sigma_x]$")
    ax.set_ylabel(r"$2^{nd}$ feature $[\sigma_y]$")
    ax.legend()
    ax.grid()
    return figure


def image_to_tensor(figure: plt.Figure) -> tf.Tensor:
    """Converts a pyplot image to a tensor.
    Useful for TensorBoard loading. Saves first the image in memory and then
    dumps to tensor.
    Parameters
    ----------
    figure: plt.Figure
        The image to be converted.
    Returns
    -------
    tf.Tensor
        The RGBA image decoded in tensor form, of shape=(1, H, W, C). (C=4)
    """
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close(figure)
    buffer.seek(0)
    image = tf.io.decode_png(buffer.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image


def save_scatterplot_features_image(
    fname: Path, features: tf.Tensor, y_true: tf.Tensor
):
    """Saves to file a scatterplot of the first two features in tensor format.
    Parameters
    ----------
    fname: Path
        The file where to save the plot.
    features: tf.Tensor
        The tensor of features shape=(nb points, nb_features).
    y_true: tf.Tensor
        The boolean tensor of ground truths of shape=(nb points,).
    """
    figure = scatterplot_features_image(features, y_true)
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close(figure)


def save_histogram_activations_image(fname: Path, y_pred: tf.Tensor, y_true: tf.Tensor):
    """Saves histogram image to file.
    Parameters
    ----------
    fname: Path
        The file where to save the plot.
    y_pred: tf.Tensor
        The tensor of predicted scores shape=(nb points,).
    y_true: tf.Tensor
        The tensor of ground truths of shape=(nb points,).
    """
    figure = histogram_activations_image(y_pred, y_true)
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close(figure)
