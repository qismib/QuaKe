""" This module implements the CNN network."""
from typing import Tuple
import tensorflow as tf
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense, Flatten, LeakyReLU, Concatenate, Input
from .layers import CBA, CBAD
from ..AbstractNet import AbstractNet


class CNN_Network(AbstractNet):
    """Class defining Convolutional Network.

    In this approach the network is trained as a binary classifier. Input are
    three 2D projections of each event in (B,H,W,C) image form.
    """

    def __init__(
        self,
        nb_features: int = 2,
        nb_bins: list[int] = [8, 8, 40],
        nb_hidden_layers: int = 2,
        hidden_units: int = 50,
        kernel_size: list = [3, 3],
        strides: list = [1, 1],
        batch_size: int = 8,
        activation: str = "relu",
        alpha: float = 0.2,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        verbose: bool = False,
        name: str = "AttentionNetwork",
        **kwargs,
    ):
        """
        Parameters
        ----------
        nb_features: int
            The number of features to be extracted.
        nb_bins: list[int]
            The number of bins in the x, y, z axes.
        nb_hidden_layers: int
            The number of convolutional layers in the network.
        hidden_units: int
            The number of convolutional filters to be applied.
        kernel_size: list
            The height and width of the 2D convolution window.
        strides: list
            The strides of the convolution along the height and width.
        batch_size: int
            The effective batch size for gradient descent.
        activation: str
            Default keras layer activation.
        alpha: float
            Leaky relu negative slope coefficient.
        dropout_rate: float
            Dropout percentage.
        use_bias: bool
            Wether to use bias or not.
        verbose: bool
            Wether to print extra training information.
        name: str
            The name of the neural network instance.
        """

        super(CNN_Network, self).__init__(name=name, **kwargs)

        # store args
        self.nb_features = nb_features
        self.nb_xbins, self.nb_ybins, self.nb_zbins = nb_bins
        self.nb_hidden_layers = nb_hidden_layers
        self.hidden_units = hidden_units
        self.kernel_size = kernel_size
        self.strides = strides
        self.batch_size = int(batch_size)
        self.activation = activation
        self.alpha = alpha
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.verbose = verbose
        # TODO: check all attributes are indeed used

        ff_layer = CBAD if self.dropout_rate else CBA
        ff_kwargs = {"rate": self.dropout_rate} if self.dropout_rate else {}

        self.yz_encoder = [
            ff_layer(
                self.hidden_units,
                self.kernel_size,
                self.strides,
                self.activation,
                self.alpha,
                name=f"yz_enc_{i}",
                **ff_kwargs,
            )
            for i in range(self.nb_hidden_layers)
        ]
        self.xz_encoder = [
            ff_layer(
                self.hidden_units,
                self.kernel_size,
                self.strides,
                self.activation,
                self.alpha,
                name=f"xz_enc_{i}",
                **ff_kwargs,
            )
            for i in range(self.nb_hidden_layers)
        ]
        self.xy_encoder = [
            ff_layer(
                self.hidden_units,
                self.kernel_size,
                self.strides,
                self.activation,
                self.alpha,
                name=f"xy_enc_{i}",
                **ff_kwargs,
            )
            for i in range(self.nb_hidden_layers)
        ]

        self.flatten = Flatten(name="flatten")
        self.cat = Concatenate(axis=-1, name="cat")
        # self.dense_0 = Dense(10, name="fc_0")
        # self.lrelu_0 = LeakyReLU(alpha=self.alpha, name="lrelu_0")

        # self.dense_1 = Dense(self.nb_features, name="fc_1")
        # self.lrelu_1 = LeakyReLU(alpha=self.alpha, name="features")
        self.dense_1 = Dense(self.nb_features, name="features")
        self.cat = Concatenate(axis=-1, name="cat2")
        self.lrelu_1 = LeakyReLU(alpha=self.alpha, name="lrelu_1")

        # self.dense_2 = Dense(2, name = "dense_2") # ultimi cambi: 10, 15, 20, 25...
        # self.lrelu_2 = LeakyReLU(alpha = self.alpha)
        self.final = Dense(1, name="final")

        # explicitly build network weights
        build_with_shape = (
            (self.nb_ybins, self.nb_zbins, 1),
            (self.nb_xbins, self.nb_zbins, 1),
            (self.nb_xbins, self.nb_ybins, 1),
        )
        names = ("yz", "xz", "xy")
        batched_shape = [(self.batch_size,) + i for i in build_with_shape]
        self.inputs_layer = [
            Input(shape=bws, name=n) for bws, n in zip(build_with_shape, names)
        ]
        super(CNN_Network, self).build(batched_shape)

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor],
        training: bool = None,
    ) -> tf.Tensor:
        """Convolutional network forward pass.

        Parameters
        ----------
        inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            yz, xz, xy, 2D projections, each of shape=(batch,H,W,C).
        training: bool
            Wether the netowrk is in training mode or not.

        Returns
        -------
        output: tf.Tensor
            Classification score of shape=(batch,).
        """

        features = self.feature_extraction(inputs, training=training)
        output = self.final(self.lrelu_1(features))
        # output = self.final(features)
        output = tf.squeeze(sigmoid(output), axis=-1)
        if self.return_features:
            return output, features
        return output

    def feature_extraction(
        self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], training: bool = None
    ) -> tf.Tensor:
        """Convolutional network feature extraction.

        Provides the forward pass for feature extraction. The downstream
        classification is independent.

        Parameters
        ----------
        inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
            yz, xz, xy, 2D projections, each of shape=(batch,H,W,C).
        training: bool
            Wether the netowrk is in training mode or not.
        return_features: bool
            Wether to return extracted features or not.

        Returns
        -------
        features: tf.Tensor
            The extracted features, of shape=(batch, nb_features).
        """
        yz_planes, xz_planes, xy_planes = inputs

        for enc in self.yz_encoder:
            yz_planes = enc(yz_planes)

        for enc in self.xz_encoder:
            xz_planes = enc(xz_planes)

        for enc in self.xy_encoder:
            xy_planes = enc(xy_planes)

        yz = self.flatten(yz_planes)
        xz = self.flatten(xz_planes)
        xy = self.flatten(xy_planes)

        feats = self.cat([yz, xz, xy])
        # x = self.lrelu_0(self.dense_0(feats))
        # features = self.dense_1(x)
        features = self.dense_1(feats)
        return features

    def train_step(self, data: list[tf.Tensor]) -> dict:
        """Overloading of the train_step method, called during model.fit.

        This function is mainly used for debugging purposes. Saves the gradients
        at each step.

        Parameters
        ----------
        data: list[tf.Tensor]
            yz, xz, xy, 2D projections, each of shape=(batch,H,W,C).

        Returns
        -------
        dict
            The updated metrics dictionary.
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value (configured in compile method)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # save gradients for debugging purposes
        self.current_gradients = gradients

        # print_gradients_to_writer(zip(gradients, self.trainable_weights))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
