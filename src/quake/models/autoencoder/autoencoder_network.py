import logging
import numpy as np
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    Dense,
    MultiHeadAttention,
    Attention,
    Concatenate,
    LeakyReLU,
    BatchNormalization,
)
from tensorflow.keras.activations import sigmoid
from quake import PACKAGE
from ..AbstractNet import AbstractNet
from .layers import (
    Head,
    LBA,
    LBAD,
    apply_random_rotation_2d,
    apply_random_rotation_3d,
)

logger = logging.getLogger(PACKAGE + ".autoencoder")


class Autoencoder(AbstractNet):
    """Class defining Autoencoder Network.

    In this approach the network is trained as an autoencoder. Input is a
    point cloud with features accompaining each node: features describe 3D hit
    position and hit energy.

    Note: the number of hits may vary, the input and output tensors are padded
    with the last hit value.
    """

    def __init__(
        self,
        f_dims: int = 4,
        spatial_dims: int = 3,
        use_spatial_dims: bool = False,
        attention_nodes: int = 1,
        nb_heads: int = 1,
        enc_filters: list = [16, 8, 4],
        dec_filters: list = [8, 16],
        batch_size: int = 8,
        activation: str = "relu",
        alpha: float = 0.2,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        verbose: bool = False,
        max_input_nb: int = None,
        name: str = "Autoencoder",
        spatial_rec_accuracy: list = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        f_dims: int
            Number of point cloud feature dimensions.
        spatial_dims: int
            Number of point cloud spatial feature dimensions.
        use_spatial_dims: bool
            Wether to use hit positions together with hit energies.
        attention_nodes: int
            Attention heads.
        enc_filders: list
            Encoding feed forward head list.
        dec_filters: list
            Deconding feed forward head list.
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
        max_input_nb: int
            Max hit number in the dataset.
        name: str
            The name of the neural network instance.
        """
        super().__init__(name=name, **kwargs)

        # store args
        self.f_dims = f_dims
        self.spatial_dims = spatial_dims
        self.attention_nodes = attention_nodes
        self.enc_filters = enc_filters
        self.dec_filters = dec_filters
        self.batch_size = int(batch_size)
        self.activation = activation
        self.alpha = alpha
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.verbose = verbose
        self.max_length = max_input_nb
        self.spatial_rec_accuracy = spatial_rec_accuracy
        self.use_spatial_dims = use_spatial_dims
        self.nb_heads = nb_heads

        self.apply_random_rotation = (
            apply_random_rotation_2d
            if self.spatial_dims == 2
            else apply_random_rotation_3d
        )

        ff_layer = LBAD if self.dropout_rate else LBA
        ff_kwargs = {"rate": self.dropout_rate} if self.dropout_rate else {}
        self.encoding = []

        # self-attention feature extraction
        self.attention = []
        for i in range(self.attention_nodes):
            self.attention.append(MultiHeadAttention(self.nb_heads, np.concatenate([enc_filters, dec_filters])[i], name = f"MHA{i}"))
            # self.attention.append(Attention(name = f"MHA{i}"))
        # encoding layers
        self.encoding = [
            ff_layer(
                units,
                self.activation,
                self.alpha,
                name=f"Enc{i}",
                **ff_kwargs,
            )
            for i, units in enumerate(enc_filters)
        ]
        # decoding layers
        self.decoding = [
            ff_layer(
                units,
                self.activation,
                self.alpha,
                name=f"Dec{i}",
                **ff_kwargs,
            )
            for i, units in enumerate(dec_filters[:-1])
        ]
        self.decoding.append(
            ff_layer(
                dec_filters[-1], None, None, name=f"Dec{len(dec_filters)}", **ff_kwargs
            )
        )


        enc_length = len(self.encoding)
        att_length = len(self.attention)
        if att_length > enc_length:
            self.attention_encoding = self.attention[:enc_length]
            self.attention_decoding = self.attention[enc_length:]
        else:
            self.attention_encoding = self.attention
            self.attention_decoding = []


        build_with_shape = (self.max_length,)
        self.final = [Dense(self.max_length, name="Final")]  # , LeakyReLU(alpha=alpha)]

        names = "pc"
        self.inputs_layer = Input(shape=build_with_shape, name=names)

        batched_shape = (self.batch_size,) + build_with_shape
        super(Autoencoder, self).build(batched_shape)

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = None,
    ) -> tf.Tensor:
        """Network forward pass.

        Parameters
        ----------
        inputs: tf.Tensor
            The input point cloud of hits of shape=(batch,[nb hits],f_dims).
        training: bool
            Wether network is in training or inference mode.

        Returns
        -------
        output: tf.Tensor
            Merging probability of shape=(batch,).
        """
        features = self.feature_extraction(inputs, training=training)
        x = features

        dec_length = len(self.decoding)
        att_length = len(self.attention_decoding)
        

        if self.attention_decoding:
            for i in range(max(att_length, dec_length)):
                if i < dec_length:
                    x = self.decoding[i](x)
                if i < att_length:
                    x = tf.expand_dims(x, axis = -1)
                    x = tf.squeeze(self.attention_decoding[i](x,x), -1)

                    # x = self.attention[i]([x,x])
        else:
            for dec_step in self.decoding:
                x = dec_step(x)

        # for dec_step in self.decoding:
        #     x = dec_step(x)

        for fin_step in self.final:
            x = fin_step(x)
        # output = self.reshape(self.final(x))
        output = x
        if self.return_features:
            return output, features
        return output

    def feature_extraction(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """This function provides the forward pass for feature extraction.

        The downstream classification is independent.
        The number of extracted feature per event is the number of neurons in
        the last Head-type layer in the network.

        Parameters
        ----------
        inputs: tf.Tensor
            The input point cloud of hits of shape=(batch,[nb hits],f_dims).
        training: bool
            Wether network is in training or inference mode.

        Returns
        -------
        features: tf.Tensor
            The tensor of extracted features, of shape=(batch, nb_features).
        """
        x = inputs
        # rotate the point cloud by a random angle to enforce the
        # if training:
        #     x = self.apply_random_rotation(x)
        # for mha, enc in zip(self.mhas, self.encoding):
        #     x = mha(x)
        #     x = enc(x)
        # max pooling results in a function symmetric wrt its inputs
        # the bottleneck is the width of the last encoding layer
        # x = tf.reduce_max(x, axis=1)
        
        enc_length = len(self.encoding)
        att_length = len(self.attention_encoding)
        

        if self.attention_encoding:
            for i in range(max(att_length,enc_length)):
                if i < enc_length:
                    x = self.encoding[i](x)
                if i < att_length:
                    x = tf.expand_dims(x, axis = -1)
                    x = tf.squeeze(self.attention_encoding[i](x,x), -1)

                    # x = self.attention[i]([x,x])
        else:
            for enc_step in self.encoding:
                x = enc_step(x)

        # for att_step in self.attention:
        #     x = tf.expand_dims(x, axis = -1)
        #     x = tf.squeeze(att_step(x, x), -1)

        # for enc_step in self.encoding:
        #     x = enc_step(x)
        return x

    def train_step(self, data: list[tf.Tensor]) -> dict:
        """Overloading of the train_step method.

        This function is called during `model.fit`. Used for debugging purposes.

        Saves the gradients at each step.

        Parameters
        ----------
        data: list[tf.Tensor]
            The batch of inputs of type [point cloud, mask].

        Returns
        -------
        dict:
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
