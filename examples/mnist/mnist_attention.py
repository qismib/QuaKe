import argparse
import numpy as np
from quake.models.attention.attention_network import AttentionNetwork
import tensorflow as tf
import tensorflow.keras.backend as tfK
from sklearn.model_selection import train_test_split
from quake.models.attention.attention_dataloading import Dataset
from quake.models.attention.attention_network import AttentionNetwork


def transform_data(images: np.ndarray, threshold: float) -> np.ndarray:
    pcs = []
    for i in range(len(images)):
        coords = np.argwhere(images[i] > threshold)
        values = images[i, coords[:, 0], coords[:, 1]]
        pos = (coords - 14) / 28
        pc = np.concatenate([pos, values.reshape(-1, 1)], axis=1)
        pcs.append(pc)
    pcs = np.array(pcs, dtype=object)
    return pcs


def main(batch_size, seed, dataset, threshold):
    # download dataset
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        nb_classes = 10
    elif dataset == "cifar100-coarse":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(
            "coarse"
        )
        nb_classes = 20
    elif dataset == "cifa100-fine":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(
            "fine"
        )
        nb_classes = 100
    else:
        raise ValueError(f"Dataset not recognized, found {dataset}")

    # greyscale the rgb images
    if len(x_train.shape) == 4:
        x_train = x_train.mean(-1)
        y_train = y_train.flatten()
        x_test = x_test.mean(-1)
        y_test = y_test.flatten()

    # normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    inputs_train, inputs_val, targets_train, targets_val = train_test_split(
        x_train, y_train, test_size=0.1, random_state=seed
    )

    train_generator = Dataset(
        transform_data(inputs_train, threshold),
        targets_train,
        batch_size,
        smart_batching=True,
        seed=seed,
    )
    val_generator = Dataset(
        transform_data(inputs_val, threshold), targets_val, batch_size
    )
    test_generator = Dataset(transform_data(x_test, threshold), y_test, batch_size)

    tfK.clear_session()

    setup = {
        "lr": 1e-3,
        "ckpt": None,
        "net_dict": {
            "f_dims": 3,
            "spatial_dims": 2,
            "nb_mha_heads": 4,
            "mha_filters": [8, 64, 128],
            "nb_fc_heads": 1,
            "fc_filters": [64, 64, nb_classes],
            "batch_size": 32,
            "activation": "relu",
            "alpha": 0.2,
            "dropout_rate": 0.0,
            "use_bias": True,
            "name": "AttentionNetwork",
        },
    }

    opt = tf.keras.optimizers.Adam(learning_rate=setup["lr"])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(name="xent")
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5"),
    ]

    # extract features and make the mnist classification
    base_net = AttentionNetwork(**setup["net_dict"])
    pc_inputs = tf.keras.Input(shape=(None, setup["net_dict"]["f_dims"]), name="pc")
    mask_inputs = tf.keras.Input(shape=(None, None), name="mask")
    inputs = [pc_inputs, mask_inputs]
    x = base_net.feature_extraction(inputs)
    outputs = tf.keras.layers.Dense(nb_classes, activation="softmax")(x)
    network = tf.keras.Model(inputs, outputs)

    network.compile(
        loss=loss,
        optimizer=opt,
        metrics=metrics,
    )

    network.fit(
        train_generator,
        epochs=2,
        validation_data=val_generator,
        shuffle=False,
    )

    network.evaluate(test_generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        help="the dataset to load",
        default="mnist",
        choices=["cifar100-fine", "cifar100-coarse", "mnist"],
    )
    parser.add_argument("--threshold", "-t", type=float, help="image threshold", default=0.95)
    args = parser.parse_args()

    batch_size = 32
    seed = 42
    main(batch_size, seed, args.dataset, args.threshold)
