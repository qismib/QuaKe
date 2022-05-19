""" This module implements CNN network building blocks. """
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer,
    BatchNormalization,
    Dropout,
    Conv2D,
)


class CBA(Layer):
    """Convolutional, batchnorm, activation layer stack."""

    def __init__(
        self,
        units: int,
        kernel_size: list = [3, 3],
        strides: list = [1, 1],
        act: str = "relu",
        alpha: float = 0.2,
        **kwargs,
    ):
        """
        Parameters
        ----------
        units: int
            Output feature dimensionality.
        kernel_size: list
            The height and width of the 2D convolution window.
        strides: list
            The strides of the convolution along the height and width.
        act: str
            Activation string.
        alpha: float
            Leaky relu negative slope coefficient.
        """
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.act = act
        self.alpha = alpha

        self.conv = Conv2D(
            self.units, self.kernel_size, self.strides, padding="same", name="conv"
        )
        self.activation = tf.keras.activations.get(self.act)
        # self.batchnorm = BatchNormalization(name="batchnorm")

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
        output: tf.Tensor
            Output tensor of shape=(B, ..., do).
        """
        x = self.conv(inputs)
        # x = self.batchnorm(x)
        if self.act == "relu":
            output = self.activation(x, alpha=self.alpha)
        else:
            output = self.activation(x)
        return output

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "act": self.act,
                "alpha": self.alpha,
            }
        )
        return config


class CBAD(CBA):
    """Convolutional, batchnorm, activation, dropout layer stack."""

    def __init__(
        self,
        units: int,
        kernel_size: list = [3, 3],
        strides: list = [1, 1],
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
        kernel_size: list
            The height and width of the 2D convolution window.
        strides: list
            The strides of the convolution along the height and width.
        act: str
            Activation string
        alpha: float
            Leaky relu negative slope coefficient.
        rate: float
            Dropout percentage.
        """
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.act = act
        self.alpha = alpha
        self.rate = rate

        super().__init__(
            self.units, self.kernel_size, self.strides, self.act, self.alpha, **kwargs
        )

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
        output: tf.Tensor
            Output tensor of shape=(B, ..., do).
        """
        x = super().call(inputs)
        output = self.dropout(x)
        return output

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"rate": self.rate})
        return config
