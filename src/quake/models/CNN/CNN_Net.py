from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D
from tensorflow.keras.layers import Conv3D, LeakyReLU


def buildCNN(s_tr, opts: dict):
    lr = opts["lr"]
    feature_number = opts["feature_number"]
    dim = opts["dim"]
    alpha = opts["alpha"]
    dropout_rate = opts["dropout_rate"]

    model = Sequential()
    if dim == 2:
        model.add(
            Conv2D(50, input_shape=(s_tr.shape[1:]), kernel_size=(2, 2), padding="same")
        )
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout_rate))

        model.add(Conv2D(50, kernel_size=(2, 2), padding="same"))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout_rate))

        model.add(Conv2D(50, kernel_size=(2, 2), padding="same"))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout_rate))

    elif dim == 3:
        model.add(
            Conv3D(
                50, input_shape=(s_tr.shape[1:]), kernel_size=(2, 2, 2), padding="same"
            )
        )
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout_rate))

        model.add(Conv3D(50, kernel_size=(2, 2, 2), padding="same"))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout_rate))

        model.add(Conv3D(50, kernel_size=(2, 2, 2), padding="same"))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout_rate))

    model.add(Flatten())

    model.add(Dense(feature_number, name="features"))
    model.add(LeakyReLU(alpha=alpha))

    model.add(Dense(2, activation="softmax"))

    intermediate_layer_model = Model(
        inputs=model.inputs, outputs=model.get_layer("features").output
    )
    model.compile(
        loss=categorical_crossentropy, optimizer=Adam(lr), metrics=["accuracy"]
    )
    return model
