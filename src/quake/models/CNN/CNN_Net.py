from tensorflow.keras import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D
from tensorflow.keras.layers import LeakyReLU, Concatenate, Input


def buildCNN(set, setup: dict):
    """
    Returns the CNN's architecture

    Parameters
    ----------
        - set: dataset: 2D projections of voxelized data
        - setup: settings dictionary
    """
    set_train = set[0]
    lr = setup["model"]["cnn"]["lr"]
    feature_number = setup["model"]["cnn"]["feature_number"]
    alpha = setup["model"]["cnn"]["alpha"]
    dropout_rate = setup["model"]["cnn"]["dropout_rate"]

    input1 = Input((set_train[0].shape[1:]))
    input2 = Input((set_train[1].shape[1:]))
    input3 = Input((set_train[2].shape[1:]))

    d1 = convblocks(input1, 2, alpha, dropout_rate)
    d2 = convblocks(input2, 2, alpha, dropout_rate)
    d3 = convblocks(input3, 2, alpha, dropout_rate)

    f1 = Flatten()(d1)
    f2 = Flatten()(d2)
    f3 = Flatten()(d3)

    concat_layer = Concatenate()([f1, f2, f3])

    dense1 = Dense(10)(concat_layer)
    relu1 = LeakyReLU(alpha=alpha)(dense1)
    dense2 = Dense(feature_number, name="features")(relu1)
    relu2 = LeakyReLU(alpha=alpha)(dense2)

    pred = Dense(2, activation="softmax")(relu2)

    model = Model(inputs=[input1, input2, input3], outputs=pred)
    model.compile(
        loss=categorical_crossentropy, optimizer=Adam(lr), metrics=["accuracy"]
    )
    model.summary()
    return model


def convblocks(input_layer, iterations, alpha, dropout_rate):
    """
    Returns sequences of convolutional-activation-dropout layers blocks

    Parameters
    ----------
        - input_layer: first layer of the Neural Network
        - iterations: number of blocks
        - alpha: ReLU leakage parameter
        - dropout_rate: dropout strength
    """
    conv = list()
    leaky = list()
    drop = list()

    for i in range(0, iterations):
        if i == 0:
            conv.append(Conv2D(50, kernel_size=(2, 2))(input_layer))
        else:
            conv.append(Conv2D(50, kernel_size=(2, 2))(drop[i - 1]))
        leaky.append(LeakyReLU(alpha=alpha)(conv[i]))
        drop.append(Dropout(dropout_rate)(leaky[i]))
    return drop[-1]
