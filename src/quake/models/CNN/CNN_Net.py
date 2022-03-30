from tensorflow.keras import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D
from tensorflow.keras.layers import LeakyReLU, Concatenate, Input


def buildCNN(s_tr, opts: dict):
    lr = opts["lr"]
    feature_number = opts["feature_number"]
    alpha = opts["alpha"]
    dropout_rate = opts["dropout_rate"]

    input1 = Input((s_tr[0].shape[1:]))
    input2 = Input((s_tr[1].shape[1:]))
    input3 = Input((s_tr[2].shape[1:]))

    d1 = convblocks(input1, 3, alpha, dropout_rate)
    d2 = convblocks(input2, 3, alpha, dropout_rate)
    d3 = convblocks(input3, 3, alpha, dropout_rate)

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
    return model


def convblocks(input_layer, iterations, alpha, dropout_rate):
    c = list()
    l = list()
    d = list()

    c.append(Conv2D(50, kernel_size=(2, 2))(input_layer))
    for i in range(0, iterations):
        if i == 0:
            c.append(Conv2D(50, kernel_size=(2, 2))(input_layer))
        else:
            c.append(Conv2D(50, kernel_size=(2, 2))(d[i - 1]))
        l.append(LeakyReLU(alpha=alpha)(c[i]))
        d.append(Dropout(dropout_rate)(l[i]))
    return d[-1]
