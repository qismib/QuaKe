import logging
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import sigmoid
from quake import PACKAGE
from .AbstractNet import AbstractNet
from .layers import TransformerEncoder, Head, LBA, LBAD

logger = logging.getLogger(PACKAGE + ".attention")


class AttentionNetwork(AbstractNet):
    """
    Class defining Attention Network.
    In this approach the network is trained as a binary classifier. Inpiut is a
    point cloud with features accompaining each node: features describe 3D hit
    position and hit energy.

    Note: the number of hits may vary, hence each batch is padded according to
    the maximum example length in the batch.
    """

    def __init__(
        self,
        f_dims: int = 4,
        nb_mha_heads: int = 2,
        mha_filters: list = [8, 16],
        nb_fc_heads: int = 2,
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
            - f_dims: number of point cloud feature dimensions
            - nb_mha_heads: the number of heads in the `MultiHeadAttention` layer
            - mha_filters: the output units for each `MultiHeadAttention` in the stack
            - nb_fc_heads: the number of `Head` layers to be concatenated
            - fc_filters: the output units for each `Head` in the stack
            - batch_size: the effective batch size for gradient descent
            - activation: default keras layer activation
            - alpha: leaky relu negative slope coefficient
            - dropout_rate: dropout percentage
            - use_bias: wether to use bias or not
            - verbose: wether to print extra training information
            - name: the name of the neural network instance
        """
        super(AttentionNetwork, self).__init__(name=name, **kwargs)

        # store args
        self.f_dims = f_dims
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

        self.mhas = []
        self.encoding = []

        fins = [self.f_dims] + self.mha_filters[:-1]
        fouts = self.mha_filters
        ff_layer = LBAD if self.dropout_rate else LBA
        ff_kwargs = {"rate": self.dropout_rate} if self.dropout_rate else {}

        for i, (fin, fout) in enumerate(zip(fins, fouts)):
            # attention layers
            self.mhas.append(
                TransformerEncoder(fin, self.nb_mha_heads, name=f"mha_{i}")
            )
            # encoding layers (responsible of changing the feature axis dimension)
            self.encoding.append(
                ff_layer(
                    fout, self.activation, self.alpha, name=f"enc_{i}", **ff_kwargs
                )
            )

        # decoding layers
        self.heads = [
            Head(
                self.fc_filters,
                self.activation,
                self.alpha,
                self.dropout_rate,
                name=f"dec_{i}",
            )
            for i in range(self.nb_fc_heads)
        ]

        self.final = Dense(1, name="final")

        # explicitly build network weights
        build_with_shape = ((None, self.f_dims), (None, None))
        names = ("pc", "mask")
        batched_shape = [(self.batch_size,) + i for i in build_with_shape]
        self.input_layer = [
            Input(shape=bws, name=n) for bws, n in zip(build_with_shape, names)
        ]
        super(AttentionNetwork, self).build(batched_shape)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        Parameters
        ----------
            - inputs
                - point cloud of hits of shape=(batch,[nb hits],f_dims)
                - mask tensor of shape=(batch,[nb hits],f_dims)
        Returns
        -------
            - merging probability of shape=(batch,)
        """
        x, mask = inputs
        for mha, enc in zip(self.mhas, self.encoding):
            x = mha(x, attention_mask=mask)
            x = enc(x)

        # TODO: think about replacing it with max-pooling and average-pooling
        # ops before reducing everything
        x = tf.reduce_max(x, axis=1)

        results = []
        for head in self.heads:
            results.append(head(x))

        output = tf.stack(results, axis=-1)
        output = tf.reduce_mean(output, axis=-1)
        output = tf.squeeze(sigmoid(self.final(output)), axis=-1)
        return output

    def overload_train_step(self, data):
        """
        Overloading of the train_step method, which is called during model.fit.
        It can be used to print low level information on gradients. Mainly used
        for debugging purposes.
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

        # debugging gradients (look if they are vanishing)
        print_gradients(zip(gradients, self.trainable_weights))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def print_gradients(gradients):
    for g, t in gradients:
        tf.print(
            "Param:",
            g.name,
            ", value:",
            tf.reduce_mean(t),
            tf.math.reduce_std(t),
            ", grad:",
            tf.reduce_mean(g),
            tf.math.reduce_std(g),
        )
    tf.print("---------------------------")
    return True
