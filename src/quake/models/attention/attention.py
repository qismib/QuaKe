import logging
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.activations import sigmoid
from quake import PACKAGE
from .AbstractNet import AbstractNet, get_activation
from .layers import TransformerEncoder, Head

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
        self.activation = get_activation(activation)
        self.use_bias = use_bias
        self.verbose = verbose

        # adapt the output if requested
        if self.fc_filters[-1] != 1:
            logger.warning(
                f"AttentionNetwork last layer must have one neuron only, but found "
                f"{self.fc_filters[-1]}: adapting last layer ..."
            )
            self.fc_filters.append(self.units)

        # attention layers
        self.mhas = [
            TransformerEncoder(dout, self.nb_mha_heads, name=f"mha_{ih}")
            for ih, dout in enumerate(self.mha_filters)
        ]

        # feed-forward layers
        self.heads = [
            Head(
                self.fc_filters,
                activation=self.activation,
                name=f"head_{ih}",
            )
            for ih in range(self.nb_fc_heads)
        ]
        self.concat = Concatenate(axis=-1, name="cat")

        # explicitly build network weights
        build_with_shape = ((None, self.f_dims), (None, None))
        names = ("pc", "mask")
        batched_shape = [(self.batch_size,) + i for i in build_with_shape]
        self.input_layer = [
            Input(shape=bws, name=n) for bws, n in zip(build_with_shape, names)
        ]
        super(AttentionNetwork, self).build(batched_shape)

    # ----------------------------------------------------------------------
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
        for mha in self.mhas:
            x = self.activation(mha(x, attention_mask=mask))

        x = tf.reduce_max(x, axis=1)

        results = []
        for head in self.heads:
            results.append(head(x))

        return tf.reduce_mean(sigmoid(self.concat(results)), axis=-1)
