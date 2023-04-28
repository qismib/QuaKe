from pathlib import Path
import numpy as np
import tensorflow as tf
from .diagnostics import (
    scatterplot_features_image,
    histogram_activations_image,
    image_to_tensor,
)
from deeplar.models.AbstractNet import FeatureReturner


class DebuggingCallback(tf.keras.callbacks.Callback):
    """Callback to register network activations and feature histograms to
    TensorBoard on epoch end.
    Prints histograms for a batch of last layer activations and
    extracted features.
    Prints gradients received by all the layers as an histogram, as well as
    scalar plot with their mean.
    """

    def __init__(self, logdir: Path, validation_data: tf.keras.utils.Sequence):
        """
        Parameters
        ----------
        logdir: Path
            Log directory.
        validation_data: tf.keras.utils.Sequence
            Validation dataset generator.
        """
        super().__init__()
        self.logdir = logdir
        self.validation_data = validation_data
        self.y_true = self.validation_data.targets
        self.writer = tf.summary.create_file_writer(self.logdir.as_posix())

    def on_epoch_begin(self, epoch, logs=None):
        self.initial_weights = [
            tf.convert_to_tensor(var)
            for layer in self.model.layers
            for var in layer.trainable_variables
        ]
        self.weights_names = [
            var.name for layer in self.model.layers for var in layer.trainable_variables
        ]

    def on_epoch_end(self, epoch, logs=None):
        """Extracts feature and final activation histograms.
        Extracts the (cumulative) gradients values trend and histograms.
        """
        # extract features and final activations
        with FeatureReturner(self.model) as fr:
            y_pred, features = fr.predict(self.validation_data, verbose=0)

        self.current_weights = [
            tf.convert_to_tensor(var)
            for layer in self.model.layers
            for var in layer.trainable_variables
        ]

        with self.writer.as_default():
            tf.summary.image(
                "Activations scores",
                tf_histogram_activations(y_pred, self.y_true),
                step=epoch,
            )
            tf.summary.image(
                "Features scatterplot",
                tf_scatterplot_features(features, self.y_true, self.model.layers[-1]),
                step=epoch,
            )

            for name, initial, current in zip(
                self.weights_names,
                self.initial_weights,
                self.current_weights,
            ):
                gradients = current - initial
                tf.summary.histogram(name[:-2] + "/gradients", gradients, step=epoch)
                tf.summary.scalar(
                    name[:-2] + "/grad_mean", tf.math.reduce_mean(gradients), step=epoch
                )

                tf.summary.histogram(name[:-2] + "/values", current, step=epoch)
                tf.summary.scalar(
                    name[:-2] + "/values_mean",
                    tf.math.reduce_mean(current),
                    step=epoch,
                )
        self.writer.flush()


def tf_histogram_activations(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    """Returns a histogram of the network classification scores in tensor format.
    Parameters
    ----------
    y_pred: tf.Tensor
        The tensor of predicted scores shape=(nb points,).
    y_true: tf.Tensor
        The boolean tensor of ground truths of shape=(nb points,).
    Returns
    -------
    tf.Tensor
        the RGBA image decoded in tensor form, of shape=(1, H, W, 4).
    """
    figure = histogram_activations_image(y_pred, y_true)
    return image_to_tensor(figure)


def tf_scatterplot_features(
    features: tf.Tensor, y_true: tf.Tensor, layer: tf.keras.layers.Layer
) -> tf.Tensor:
    """Returns a scatterplot of the first two features in tensor format.
    Parameters
    ----------
    features: tf.Tensor
        The tensor of features shape=(nb points, nb_features).
    y_true: tf.Tensor
        The boolean tensor of ground truths of shape=(nb points,).
    Returns
    -------
    tf.Tensor
        The RGBA image decoded in tensor form, of shape=(1, H, W, 4).
    """
    figure = scatterplot_features_image(features, y_true)

    # print decision line
    pts = np.array([-3.5, 3.5])
    ax = figure.axes[0]
    kernel, bias = layer.weights
    if kernel[1] != 0:
        m = (-1.0 * kernel[0] / kernel[1]).numpy().flatten()
        q = (-1.0 * bias / kernel[1]).numpy().flatten()
        y_final = m * pts + q
        # print(f"Decision line: y = {m:.3f} x + {q:.3f}")
        msg = f"Decision line: y = {m[0]:.3f} x {'+' if np.sign(q) else '-'} {np.abs(q[0]):.3f}"
    else:
        if kernel[0] != 0:
            q = (-1.0 * bias / kernel[1]).numpy().flatten()
            msg = f"Vertical decision line at x = {q}"
            pts = [q, q]
            y_final = [-3.5, 3.5]
        else:
            msg = "All weights are zero !"
            pts = []
            y_final = []
    ax.title.set_text(f"Standardized network extracted features\n{msg}")
    ax.plot(pts, y_final, lw=0.5, c="blue", linestyle="dashed")

    return image_to_tensor(figure)
