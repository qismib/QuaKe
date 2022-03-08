import tensorflow as tf
from tensorflow.keras.models import Model
from quake import PACKAGE


class AbstractNet(Model):
    """
    Network abstract class.

    The daughter class must define the `input_layer` attribute to use the model
    method. This should contain the output of tf.keras.Input function.
    """

    def __init__(self, name: str, **kwargs):
        self.inputs_layer = None
        super(AbstractNet, self).__init__(name=name, **kwargs)

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
