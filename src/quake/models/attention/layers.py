""" This module implements the attention network building blocks. """
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    LayerNormalization,
    BatchNormalization,
    Dropout,
    Dense,
    MultiHeadAttention,
)


class LBA(Layer):
    """Linear, batchnorm, activation layer stack."""

    def __init__(self, units: int, act: str = "relu", alpha: float = 0.2, **kwargs):
        """
        Parameters:
        units: int
            Output feature dimensionality.
        act: str
            Activation string.
        alpha: float
            Leaky relu negative slope coefficient.
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
        inputs: tf.Tensor
            Input tensor of shape=(B, ..., d_in).

        Returns
        -------
        tf.Tensor
            Output tensor of shape=(B, ..., do).
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
        Parameters
        ----------
        units: int
            Output feature dimensionality.
        act: str
            Activation string
        alpha: float
            Leaky relu negative slope coefficient.
        rate: float
            Dropout percentage.
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
        tf.Tensor
            inputs: input tensor of shape=(B, ..., d_in).

        Returns
        -------
        tf.Tensor
            Output tensor of shape=(B, ..., do).
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
