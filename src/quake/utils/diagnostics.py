""" This module contains utility function to make diagnostics plots. """
import io
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from quake.utils.configflow import bool_me
from sklearn.manifold import TSNE


def histogram_activations_image(y_pred: tf.Tensor, y_true: tf.Tensor) -> plt.Figure:
    """
    Returns a histogram of the network classification scores.
    Overlapped histograms scores for the positive and negative samples.

    Parameters
    ----------
        - y_pred: the tensor of predicted scores shape=(nb points,)
        - y_true: the tensor of ground truths of shape=(nb points,)

    Returns
    -------
        - the pyplot histograms
    """
    y_pred = y_pred.numpy()
    y_true = bool_me(y_true).numpy()

    scores_true = y_pred[y_true]
    scores_false = y_pred[~y_true]

    tp = np.count_nonzero(scores_true > 0.5)
    fp = np.count_nonzero(scores_false < 0.5)
    tot = len(y_pred)
    accuracy = (tp + fp) / tot

    bins = np.linspace(0, 1, 101)
    h_true, _ = np.histogram(scores_true, bins=bins)
    h_false, _ = np.histogram(scores_false, bins=bins)
    figure = plt.figure()
    ax = figure.add_subplot()
    ax.title.set_text(f"Network classification. Accuracy: {accuracy*100:.3f}%")
    ax.hist(
        bins[:-1],
        bins,
        weights=h_true,
        histtype="step",
        lw=0.5,
        color="green",
        label=r"$0\nu\beta\beta$",
    )
    ax.hist(
        bins[:-1],
        bins,
        weights=h_false,
        histtype="step",
        lw=0.5,
        color="red",
        label=r"$e^-$",
    )
    ax.set_xlabel("Network score")
    ax.legend()
    return figure


def scatterplot_features_image(features: tf.Tensor, y_true: tf.Tensor) -> plt.Figure:
    """
    Returns a 2D scatterplot of the input features.
    Useful to look visually the separation in the 2D plane for binary
    classification.
    The input features should have shape `(nb points, nb dims)`.
    If `nb dims == 2`, plot features in 2D plane.
    If `nb dims == 3`, plot features in 3D space.
    If `nb dims > 3`, use tSNE technique to reduce point to 2 dimensions and
    plot into 2D plane.

    Parameters
    ----------
        - features: the tensor of features shape=(nb points, nb_features)
        - y_true: the tensor of ground truths of shape=(nb points,)

    Returns
    -------
        - the pyplot scatterplot
    """
    features = features.numpy()
    features = (features - features.mean(0, keepdims=True)) / features.std(0, keepdims=True)
    nb_dims = features.shape[1]
    
    y_true = bool_me(y_true).numpy()

    projection = '3d' if nb_dims == 3 else None # use 3D space only if nb_dims is 3
    # define scatter function arguments depending on point dimensionality
    if 1 < nb_dims <= 3:
        pos = features[y_true]
        neg = features[~y_true]
        if nb_dims == 2:
            pos_args = (pos[:,0], pos[:,1])
            neg_args = (neg[:,0], neg[:,1])
        if nb_dims == 3:
            pos_args = (pos[:,0], pos[:,1], pos[:,2])
            neg_args = (neg[:,0], neg[:,1], neg[:,2])
    elif nb_dims > 3:
        transformed = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(features)
        pos = transformed[y_true]
        neg = transformed[~y_true]
        pos_args = (pos[:,0], pos[:,1])
        neg_args = (neg[:,0], neg[:,1])
    else:
        raise ValueError(f"The extracted features must be more than 1, got {nb_dims}")

    figure = plt.figure()
    ax = figure.add_subplot(projection=projection)
    ax.title.set_text("Standardized network extracted features")
    ax.scatter(*pos_args, c="green", s=10, label=r"$0\nu\beta\beta$")
    ax.scatter(*neg_args, c="red", s=10, label=r"$e^-$")
    ax.set_xlabel(r"$1^{st}$ feature $[\sigma_x]$")
    ax.set_ylabel(r"$2^{nd}$ feature $[\sigma_y]$")
    ax.legend()
    ax.grid()
    return figure


def image_to_tensor(figure: plt.Figure) -> tf.Tensor:
    """
    Converts a pyplot image to a tensor. Useful for TensorBoard loading.
    Saves first the image in memory and then dumps to tensor.

    Parameters
    ----------
        - figure: the image to be converted

    Returns
    -------
        - the RGBA image decoded in tensor form, of shape=(1, H, W, C). (C=4)
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
    """
    Saves to file a scatterplot of the first two features in tensor format.

    Parameters
    ----------
        - fname: the file where to save the plot
        - features: the tensor of features shape=(nb points, nb_features)
        - y_true: the boolean tensor of ground truths of shape=(nb points,)
    """
    figure = scatterplot_features_image(features, y_true)
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close(figure)


def save_histogram_activations_image(fname: Path, y_pred: tf.Tensor, y_true: tf.Tensor):
    """
    Saves to file
    Returns a scatterplot of the first two features in tensor format.

    Parameters
    ----------
        - fname: the file where to save the plot
        - y_pred: the tensor of predicted scores shape=(nb points,)
        - y_true: the tensor of ground truths of shape=(nb points,)
    """
    figure = histogram_activations_image(y_pred, y_true)
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close(figure)