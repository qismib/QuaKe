import logging
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import sigmoid
from deeplar import PACKAGE
from ..AbstractNet import AbstractNet
from .layers import (
    TransformerEncoder,
    Head,
    LBA,
    LBAD,
    apply_random_rotation_2d,
    apply_random_rotation_3d,
)

logger = logging.getLogger(PACKAGE + ".attention")


class AttentionNetwork(AbstractNet):
    """Class defining Attention Network.

    In this approach the network is trained as a binary classifier. Inpiut is a
    point cloud with features accompaining each node: features describe 3D hit
    position and hit energy.

    Note: the number of hits may vary, hence each batch is padded according to
    the maximum example length in the batch.
    """

    def __init__(
        self,
        f_dims: int = 4,
        spatial_dims: int = 3,
        nb_mha_heads: int = 2,
        mha_filters: list = [8, 16],
        nb_fc_heads: int = 3,
        fc_filters: list = [16, 8, 4, 2, 1],
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
        f_dims: int
            Number of point cloud feature dimensions.
        spatial_dims: int
            Number of point cloud spatial feature dimensions.
        nb_mha_heads: int
            The number of heads in the `MultiHeadAttention` layer.
        mha_filters: list
            The output units for each `MultiHeadAttention` in the stack.
        nb_fc_heads: int
            The number of `Head` layers to be concatenated.
        fc_filters: list
            The output units for each `Head` in the stack.
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
        super().__init__(name=name, **kwargs)

        # store args
        self.f_dims = f_dims
        self.spatial_dims = spatial_dims
        self.nb_mha_heads = nb_mha_heads
        self.mha_filters = mha_filters
        self.nb_fc_heads = nb_fc_heads
        self.fc_filters = fc_filters
        self.batch_size = int(batch_size)
        self.activation = activation
        self.alpha = alpha
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.verbose = verbose

        self.apply_random_rotation = (
            apply_random_rotation_2d
            if self.spatial_dims == 2
            else apply_random_rotation_3d
        )

        self.mhas = []
        self.encoding = []

        fins = [self.f_dims] + self.mha_filters[:-1]
        fouts = self.mha_filters
        ff_layer = LBAD if self.dropout_rate else LBA
        ff_kwargs = {"rate": self.dropout_rate} if self.dropout_rate else {}

        for i, (fin, fout) in enumerate(zip(fins, fouts)):
            # attention layers
            self.mhas.append(TransformerEncoder(fin, self.nb_mha_heads, name=f"Mha{i}"))
            # encoding layers (responsible of changing the feature axis dimension)
            self.encoding.append(
                ff_layer(fout, self.activation, self.alpha, name=f"Enc{i}", **ff_kwargs)
            )

        # decoding layers
        self.heads = [
            Head(
                self.fc_filters,
                self.activation,
                self.alpha,
                self.dropout_rate,
                name=f"Dec{i}",
            )
            for i in range(self.nb_fc_heads)
        ]

        self.final = Dense(1, name="Final")

        # explicitly build network weights
        build_with_shape = (None, self.f_dims)
        names = "pc"
        batched_shape = (self.batch_size,) + build_with_shape
        self.inputs_layer = Input(shape=build_with_shape, name=names)
        super().build(batched_shape)

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

        output = self.final(features)
        output = tf.squeeze(sigmoid(output), axis=-1)

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
        for mha, enc in zip(self.mhas, self.encoding):
            x = mha(x)
            x = enc(x)
        # max pooling results in a function symmetric wrt its inputs
        # the bottleneck is the width of the last encoding layer
        x = tf.reduce_max(x, axis=1)

        results = []
        for head in self.heads:
            # if self.return_features:
            # #     import pdb; pdb.set_trace()
            #     head.activation = None
            results.append(head(x))
        output = tf.stack(results, axis=-1)

        output = tf.reduce_mean(output, axis=-1)
        return output

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
