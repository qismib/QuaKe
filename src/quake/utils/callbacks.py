from pathlib import Path
import tensorflow as tf
from .diagnostics import (
    scatterplot_features_image,
    histogram_activations_image,
    image_to_tensor,
)
from quake.models.AbstractNet import FeatureReturner


class DebuggingCallback(tf.keras.callbacks.Callback):
    """
    Callback to register network activations and feature histograms to
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
            - logdir: log directory
            - validation_data: validation dataset generator
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
        """
        Extracts feature and final activation histograms.
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
                tf_scatterplot_features(features, self.y_true),
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
    """
    Returns a histogram of the network classification scores in tensor format.

    Parameters
    ----------
        - y_pred: the tensor of predicted scores shape=(nb points,)
        - y_true: the boolean tensor of ground truths of shape=(nb points,)

    Returns
    -------
        - the RGBA image decoded in tensor form, of shape=(1, H, W, C). (C=4)
    """
    figure = histogram_activations_image(y_pred, y_true)
    return image_to_tensor(figure)


def tf_scatterplot_features(features: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
    """
    Returns a scatterplot of the first two features in tensor format.

    Parameters
    ----------
        - features: the tensor of features shape=(nb points, nb_features)
        - y_true: the boolean tensor of ground truths of shape=(nb points,)

    Returns
    -------
        - the RGBA image decoded in tensor form, of shape=(1, H, W, C). (C=4)
    """
    figure = scatterplot_features_image(features, y_true)
    return image_to_tensor(figure)
