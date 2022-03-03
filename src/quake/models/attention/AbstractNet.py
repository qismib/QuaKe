import tensorflow as tf
from tensorflow.keras.models import Model
from slicerl import PACKAGE


class AbstractNet(Model):
    """
    Network abstract class.

    The daughter class must define the `input_layer` attribute to use the model
    method. This should contain the output of tf.keras.Input function.
    """

    def __init__(self, name, **kwargs):
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


def add_extension(name):
    """
    Adds extension to cumulative variables while looping over it.

    Parameters
    ----------
        - name: str, the weight name to be extended

    Returns
    -------
        - str, the extended weight name
    """
    ext = "_cum"
    l = name.split(":")
    l[-2] += ext
    return ":".join(l[:-1])


class BatchCumulativeNetwork(Model):
    """
    Network implementing gradient aggregation over mini-batch. Inheriting from
    this class overrides the Model.train_sted method, see the train_step method
    docstring.

    The daughter class must define
    """

    def build(self, input_shape):
        """
        Builds network weights and define `tf.Variable` placeholders for
        cumulative gradients.
        """
        super(BatchCumulativeNetwork, self).build(input_shape)
        self.cumulative_gradients = [
            tf.Variable(
                tf.zeros_like(this_var),
                trainable=False,
                name=add_extension(this_var.name),
            )
            for this_var in self.trainable_variables
        ]
        self.cumulative_counter = tf.Variable(
            tf.constant(0), trainable=False, name="cum_counter"
        )

    def reset_cumulator(self):
        """
        Reset counter and gradients cumulator gradients.
        """

        for i in range(len(self.cumulative_gradients)):
            self.cumulative_gradients[i].assign(
                tf.zeros_like(self.trainable_variables[i])
            )
        self.cumulative_counter.assign(tf.constant(1))

    def increment_counter(self):
        """
        Reset counter and gradients cumulator gradients.
        """
        self.cumulative_counter.assign_add(tf.constant(1))

    def train_step(self, data):
        """
        The network accumulates the gradients according to batch size, to allow
        gradient averaging over multiple inputs. The aim is to reduce the loss
        function fluctuations.
        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        if self.verbose:
            reset_2 = (self.cumulative_counter + 1) % self.batch_size == 0
            tf.cond(
                reset_2,
                lambda: print_loss(x, y, y_pred, loss),
                lambda: False,
            )

        reset = self.cumulative_counter % self.batch_size == 0
        tf.cond(reset, self.reset_cumulator, self.increment_counter)

        # Compute gradients
        trainable_vars = self.trainable_variables

        gradients = tape.gradient(loss, trainable_vars)
        for i, grad in enumerate(gradients):
            self.cumulative_gradients[i].assign_add(grad / self.batch_size)

        # Update weights
        reset = self.cumulative_counter % self.batch_size == 0

        if self.verbose:
            reset_1 = (self.cumulative_counter - 1) % self.batch_size == 0
            tf.cond(
                reset_1,
                lambda: print_gradients(zip(self.cumulative_gradients, trainable_vars)),
                lambda: False,
            )

        tf.cond(
            reset,
            lambda: self.optimizer.apply_gradients(
                zip(self.cumulative_gradients, trainable_vars)
            ),
            lambda: False,
        )

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def print_loss(x, y, y_pred, loss):
    # tf.print(
    #     ", x:",
    #     tf.reduce_mean(x),
    #     tf.math.reduce_std(x),
    #     ", y:",
    #     y,
    #     ", y_pred:",
    #     y_pred,
    #     ", loss:",
    #     loss,
    # )
    tf.print(y, y_pred)
    return True


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


def get_activation(act):
    """Get activation from string."""
    try:
        fn = tf.keras.activations.get(act)
        activation = lambda x: fn(x)
    except:
        if act == "lrelu":
            activation = lambda x: tf.keras.activations.relu(x, alpha=0.0)
        else:
            raise ValueError(f"activation not recognized by keras, found {act}")
    return activation
