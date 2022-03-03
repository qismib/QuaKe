""" This module implements the attention network building blocks. """
from typing import Callable
import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dense, MultiHeadAttention
from tensorflow.keras.activations import relu


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

        self.norm0 = LayerNormalization(axis=-1, name="ln_0")

        self.fc0 = Dense(units, activation="relu", name="mlp_0")
        self.fc1 = Dense(units, activation="relu", name="mlp_1")

        self.norm1 = LayerNormalization(axis=-1, name="ln_1")

    # ----------------------------------------------------------------------
    def build(self, input_shape):
        units = input_shape[-1]
        self.mha = MultiHeadAttention(self.mha_heads, units, name="mha")
        super(TransformerEncoder, self).build(input_shape)

    # ----------------------------------------------------------------------
    def call(self, x: tf.Tensor, attention_mask: tf.Tensor = None) -> tf.Tensor:
        """
        Parameters
        ----------
            - x: input tensor of shape=(B, L, d_in)
            - attention_mask: masking tensor of shape=(B, L, L)
        Returns
        -------
            - output tensor of shape=(B, L, d_out)
        """
        x += self.mha(x, x, attention_mask=attention_mask)
        x = self.norm0(x)
        x = self.fc1(self.fc0(x))
        output = self.norm1(x)
        return output

    # ----------------------------------------------------------------------
    def get_config(self) -> dict:
        return {"units": self.units, "mha_heads": self.mha_heads}


class Head(Layer):
    """Implementation of stacking of feed-forward layers."""

    def __init__(
        self,
        filters: list,
        dropout_idxs: list = None,
        dropout: float = None,
        activation: Callable = relu,
        kernel_initializer: str = "GlorotUniform",
        name: str = "head",
        **kwargs,
    ):
        """
        Parameters
        ----------
            - filters: the number of filters for each dense layer
            - dropout_idxs: the layers number to insert dropout
            - dropout: the dropout percentage
            - activation: default keras layer activation
            - kernel_initializer: the layer initializer string
            - name: the layer name
        """
        super(Head, self).__init__(name=name, **kwargs)
        self.filters = filters
        self.dropout_idxs = dropout_idxs
        self.dropout = dropout
        self.activation = activation
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        lyrs = []

        for i, filters in enumerate(self.filters):
            lyrs.append(
                Dense(
                    filters,
                    activation=self.activation,
                    kernel_initializer=self.kernel_initializer,
                    name=f"dense_{i}",
                )
            )

            # if i in self.dropout_idxs:
            #     pass
            #     # lyrs.append(BatchNormalization(name=f"bn_{self.nb_head}_{i}"))
            #     # lyrs.append(Dropout(self.dropout, name=f"do_{self.nb_head}_{i}"))

        self.fc = lyrs

    # ----------------------------------------------------------------------
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
        for l in self.fc:
            x = l(x)
        return x

    # ----------------------------------------------------------------------
    def get_config(self) -> dict:
        config = super(Head, self).get_config()
        config.update(
            {
                "dropout": self.dropout,
                "activation": self.activation,
            }
        )
        return config
