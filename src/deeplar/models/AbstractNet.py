import logging
from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from deeplar import PACKAGE

logger = logging.getLogger(PACKAGE)


class AbstractNet(Model):
    """
    Network abstract class for feature extraction.

    The daughter class must define the `input_layer` attribute to use the model
    method. This should contain the output of tf.keras.Input function.
    """

    def __init__(self, name: str, **kwargs):
        self.inputs_layer = None
        super(AbstractNet, self).__init__(name=name, **kwargs)
        self.__return_features = False
        self.set_attribute_in_with_statement = False

    @property
    def return_features(self) -> bool:
        """Wether network should return features or not."""
        return self.__return_features

    @return_features.setter
    def return_features(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(
                f"`return_features value must be a bool, got {type(value)} instead"
            )
        if not self.set_attribute_in_with_statement:
            logger.warning(
                "Preferred way to set the `return_features` property is "
                "calling `model.predict` function inside a `with` statement. "
                "For more information, refer to the `FeatureReturner` class implementation."
            )
        self.__return_features = value

    def model(self):
        if self.inputs_layer is not None:
            ValueError(
                "AbstractNet daughter class missed to override the inputs_layer attribute"
            )
        return Model(
            inputs=self.inputs_layer,
            outputs=self.call(self.inputs_layer),
            name=self.name,
        )


class FeatureReturner:
    """Network wrapper, to ensure features are properly returned on predict.

    The feature returner ensures that the network returns the features and the
    predicted outputs during inference.

    Example
    -------

    Preferred way to make predictions from a network, while returning
    classification outputs and extracted features:
    >> with FeatureReturner(network) as fr:
    ...    outputs, features = fr.predict(generator)

    """

    def __init__(self, network: tf.keras.models.Model):
        """
        Parameters
        ----------
        network: tf.keras.models.Model
            The neural network.
        """
        self.model = network

    def __enter__(self):
        """Sets the `return_features` attribute to `True`."""
        self.model.set_attribute_in_with_statement = True
        self.model.return_features = True
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Sets back the `return_features` attribute to `False`."""
        self.model.return_features = False
        self.model.set_attribute_in_with_statement = True

    def predict(
        self, generator: tf.keras.utils.Sequence, **kwargs: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Makes inference on all the data contained in `generator`.

        Returns both the classification scores and the extracted features.

        Parameters
        ----------
        generator: tf.keras.utils.Sequence
            The dataset generator.
        **kwargs: dict
            Extra keyword arguments to be passed to `model.predict` function.

        Returns
        -------
        outputs: np.ndarray
            The network prediction tensor, of shape=(nb events,).
        features: np.ndarray
            The extrated features tensor, of shape=(nb events, nb_features).
        """
        outputs, features = self.model.predict(generator, **kwargs)
        return outputs, features
