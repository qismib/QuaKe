""" This module implements the attention network building blocks. """
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    BatchNormalization,
    Dropout,
    Dense,
    MultiHeadAttention,
)
from quake.utils.configflow import TF_PI


class LBA(Layer):
    """Linear, batchnorm, activation layer stack."""

    def __init__(self, units: int, act: str = "relu", alpha: float = 0.2, **kwargs):
        """
        Parameters:
            - units: output feature dimensionality
            - act: activation string
            - alpha: leaky relu negative slope coefficient
        """
        super().__init__(**kwargs)
        self.units = units
        self.act = act
        self.alpha = alpha

        self.linear = Dense(self.units, name="linear")
        self.activation = tf.keras.activations.get(self.act)
        self.batchnorm = BatchNormalization(name="batchnorm")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
            - inputs: input tensor of shape=(B, ..., d_in)

        Returns
        -------
            - output tensor of shape=(B, ..., do)
        """
        x = self.linear(inputs)
        x = self.batchnorm(x)
        if self.act == "relu":
            x = self.activation(x, alpha=self.alpha)
        else:
            x = self.activation(x)
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"units": self.units, "act": self.act, "alpha": self.alpha})
        return config


class LBAD(LBA):
    """Linear, batchnorm, activation, dropout layer stack."""

    def __init__(
        self,
        units: int,
        act: str = "relu",
        alpha: float = 0.2,
        rate: float = 0.1,
        **kwargs,
    ):
        """
        Parameters:
            - units: output feature dimensionality
            - act: activation string
            - alpha: leaky relu negative slope coefficient
            - rate: dropout percentage
        """
        self.units = units
        self.act = act
        self.alpha = alpha
        self.rate = rate

        super().__init__(self.units, self.act, self.alpha, **kwargs)

        self.dropout = Dropout(self.rate, name="dropout")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Parameters
        ----------
            - inputs: input tensor of shape=(B, ..., d_in)

        Returns
        -------
            - output tensor of shape=(B, ..., do)
        """
        x = super().call(inputs)
        x = self.dropout(x)
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"rate": self.rate})
        return config


class TransformerEncoder(Layer):
    """
    Implementation of ViT Encoder layer. This block exploits the fast
    implementation of the Attention mechanism for better memory management.
    """

    def __init__(
        self,
        units: int,
        mha_heads: int,
        **kwargs,
    ):
        """
        Parameters
        ----------
            - units: output feature dimensionality
            - mha_heads: number of heads in MultiHeadAttention layers
        """
        super(TransformerEncoder, self).__init__(**kwargs)
        self.units = units
        self.mha_heads = mha_heads

        self.mha = MultiHeadAttention(self.mha_heads, self.units, name="mha")

        # self.norm0 = LayerNormalization(axis=-1, name="ln_0")
        self.norm0 = BatchNormalization(axis=-1, name="bn_0")

        self.fc0 = Dense(units, activation="relu", name="mlp_0")
        self.fc1 = Dense(units, activation="relu", name="mlp_1")

        # self.norm1 = LayerNormalization(axis=-1, name="ln_1")
        self.norm1 = BatchNormalization(axis=-1, name="bn_1")

    def build(self, input_shape):
        super(TransformerEncoder, self).build(input_shape)

    def call(self, x: tf.Tensor, attention_mask: tf.Tensor = None) -> tf.Tensor:
        """
        Parameters
        ----------
            - x: input tensor of shape=(B, L, d_in)
            - attention_mask: masking tensor of shape=(B, L, L)
        Returns
        -------
            - output tensor of shape=(B, L, d_in)
        """
        x += self.mha(x, x, attention_mask=attention_mask)
        x = self.norm0(x)
        x += self.fc1(self.fc0(x))
        output = self.norm1(x)
        return output

    def get_config(self) -> dict:
        return {"units": self.units, "mha_heads": self.mha_heads}


class Head(Layer):
    """Stack of feed-forward layers."""

    def __init__(
        self,
        filters: list,
        activation: str = "relu",
        alpha: float = 0.2,
        dropout_rate: float = None,
        name: str = "head",
        **kwargs,
    ):
        """
        Parameters
        ----------
            - filters: the number of filters for each dense layer
            - activation: layer activation
            - name: the layer name
            - dropout_rate: the dropout percentage
        """
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.alpha = alpha

        ff_layer = LBAD if self.dropout_rate else LBA
        ff_kwargs = {"rate": self.dropout_rate} if self.dropout_rate else {}

        self.ff = [
            ff_layer(filters, self.activation, self.alpha, name=f"ff_{i}", **ff_kwargs)
            for i, filters in enumerate(self.filters)
        ]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Layer forward pass.
        Parameters
        ----------
            - x : inputs of shape=(B,N,K,di)
        Returns
        -------
            - output tensor of shape=(B,N,K,do)
        """
        for l in self.ff:
            x = l(x)
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "alpha": self.alpha,
            }
        )
        return config


def get_batched_rotations(angles: tf.Tensor) -> tf.Tensor:
    """
    Returns the (3+1)d spatial rotation given batched arrays of angles. Fourth axis
    remains unchanged as it contains hit energies.
    Parameters
    ----------
        - angles: batch of three rotation angles of shape=(batch_size, 3)
    Returns
    -------
        - batch of (3+1)d rotations of shape=(batch_size, 4, 4)
    """
    cos = tf.math.cos(angles)
    sin = tf.math.sin(angles)
    zeros = tf.zeros_like(angles[:, 0])
    ones = tf.ones_like(angles[:, 0])
    rot = [0.0] * 4
    rot[0] = tf.stack(
        [
            cos[:, 0] * cos[:, 1],
            cos[:, 0] * sin[:, 1] * sin[:, 2] - sin[:, 0] * cos[:, 2],
            cos[:, 0] * sin[:, 1] * cos[:, 2] + sin[:, 0] * sin[:, 2],
            zeros,
        ],
        axis=-1,
    )
    rot[1] = tf.stack(
        [
            sin[:, 0] * cos[:, 1],
            sin[:, 0] * sin[:, 1] * sin[:, 2] + cos[:, 0] * cos[:, 2],
            sin[:, 0] * sin[:, 1] * cos[:, 2] - cos[:, 0] * sin[:, 2],
            zeros,
        ],
        axis=-1,
    )
    rot[2] = tf.stack(
        [-sin[:, 1], cos[:, 1] * sin[:, 2], cos[:, 1] * cos[:, 2], zeros], axis=-1
    )
    rot[3] = tf.stack([zeros, zeros, zeros, ones], axis=-1)
    rot = tf.stack(rot, axis=1)
    return rot


def apply_random_rotation(pc: tf.Tensor) -> tf.Tensor:
    """
    Rotates inputs in the 3D space. This rigid transformations ensure that the
    network learns the geometric structure of the given point cloud, rather than
    memorizing the point positions.
    This function draws two angles in the [0,pi]x[0,2*pi] space.
    Parameters
    ----------
        - pc: point cloud of shape=(B, max len, nb feats). Three feats are the
              xyz coordinates, while the last one is the energy and should not
              be modified
    Returns
    -------
        - the rotated point cloud
    """
    batch_size = tf.shape(pc)[0]
    angles = tf.random.uniform([batch_size, 3], maxval=2 * TF_PI)
    rot = get_batched_rotations(angles)
    rot_pc = tf.einsum("ijk,ilk->ijl", pc, rot)
    return rot_pc
