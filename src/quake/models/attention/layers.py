""" This module implements the attention network building blocks. """

import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dense, MultiHeadAttention


class TransformerEncoder(Layer):
    """
    Implementation of ViT Encoder layer. This block exploits the fast
    implementation of the Attention mechanism for better memory management.
    """

    def __init__(
        self,
        units,
        mha_heads,
        **kwargs,
    ):
        """
        Parameters
        ----------
            - units: int, output feature dimensionality
            - mha_heads: int, number of heads in MultiHeadAttention layers
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
        """
        Parameters
        ----------
        """
        units = input_shape[-1]
        self.mha = MultiHeadAttention(self.mha_heads, units, name="mha")
        super(TransformerEncoder, self).build(input_shape)

    # ----------------------------------------------------------------------
    def call(self, x, attention_mask=None):
        """
        Parameters
        ----------
            - x: tf.Tensor, input tensor of shape=(B, L, d_in)
            - attention_mask: tf.Tensor, masking tensor of shape=(B, L, L)
        Returns
        -------
            - tf.Tensor, output tensor of shape=(B, L, d_out)
        """
        x += self.mha(x, x, attention_mask=attention_mask)
        x = self.norm0(x)
        x = self.fc1(self.fc0(x))
        output = self.norm1(x)
        return output

    # ----------------------------------------------------------------------
    def get_config(self):
        return {"units": self.units, "mha_heads": self.mha_heads}


class Head(Layer):
    """Implementation of stacking of feed-forward layers."""

    def __init__(
        self,
        filters,
        dropout_idxs=None,
        dropout=None,
        activation="relu",
        kernel_initializer="GlorotUniform",
        name="head",
        **kwargs,
    ):
        """
        Parameters
        ----------
            - filters: list, the number of filters for each dense layer
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
    def call(self, x):
        """
        Layer forward pass.
        Parameters
        ----------
            - x : list of input tf.Tensors
        Returns
        -------
            - tf.Tensor of shape=(B,N,K,do)
        """
        for l in self.fc:
            x = l(x)
        return x

    # ----------------------------------------------------------------------
    def get_config(self):
        config = super(Head, self).get_config()
        config.update(
            {
                "dropout": self.dropout,
                "activation": self.activation,
            }
        )
        return config
