import logging
from typing import Tuple
from time import time as tm
from pathlib import Path
import tensorflow as tf
import tensorflow.keras.backend as tfK
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)
from .attention_dataloading import read_data, Dataset
from .attention_network import AttentionNetwork
from quake import PACKAGE

logger = logging.getLogger(PACKAGE + ".attention")


def load_and_compile_network(
    msetup: dict, run_tf_eagerly: bool = False
) -> AttentionNetwork:
    """
    Loads and compiles attention network.

    Parameters
    ----------
        - msetup: attention model settings dictionary
        - run_tf_eagerly: wether to run tf eagerly, for debugging purposes

    Returns
    -------
        - the compiled network
    """
    lr = msetup["lr"]
    opt_kwarg = {"clipvalue": 0.5}
    if msetup["optimizer"].lower() == "adam":
        opt = Adam(learning_rate=lr, **opt_kwarg)
    elif msetup["optimizer"].lower() == "sgd":
        opt = SGD(learning_rate=lr, **opt_kwarg)
    elif msetup["optimizer"].lower() == "rmsprop":
        opt = RMSprop(learning_rate=lr, **opt_kwarg)
    elif msetup["optimizer"].lower() == "adagrad":
        opt = Adagrad(learning_rate=lr, **opt_kwarg)

    network = AttentionNetwork(**msetup["net_dict"])
    loss = tf.keras.losses.BinaryCrossentropy(name="xent")
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="acc"),
        tf.keras.metrics.Precision(name="prec"),
        tf.keras.metrics.Recall(name="rec"),
    ]

    network.compile(
        loss=loss,
        optimizer=opt,
        metrics=metrics,
        run_eagerly=run_tf_eagerly,
    )

    checkpoint_filepath = msetup["ckpt"]
    if checkpoint_filepath:
        logger.info(f"Loading weights at {checkpoint_filepath}")
        network.load_weights(checkpoint_filepath)
    return network


def train_network(
    msetup: dict,
    output: Path,
    network: AttentionNetwork,
    generators: Tuple[Dataset, Dataset],
) -> AttentionNetwork:
    """
    Trains the network.

    Parameters
    ----------
        - msetup: attention model settings dictionary
        - output: the output folder
        - network: the network to be trained
        - generators: the train and validation generators

    Returns
    -------
        - the trained network
    """
    train_generator, val_generator = generators

    logdir = output / f"logs/{tm()}"
    checkpoint_filepath = output.joinpath("network.h5").as_posix()
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            save_best_only=True,
            mode="max",
            monitor="val_acc",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_acc",
            factor=0.75,
            mode="max",
            verbose=1,
            patience=msetup["reducelr_patience"],
            min_lr=msetup["min_lr"],
        ),
        TensorBoard(
            log_dir=logdir,
            # write_graph=True,
            # write_images=True,
            # update_freq='batch',
            # histogram_freq=5,
            # profile_batch=5,
        ),
    ]
    if msetup["es_patience"]:
        callbacks.append(
            EarlyStopping(
                monitor="val_acc",
                min_delta=0.0001,
                mode="max",
                patience=msetup["es_patience"],
                restore_best_weights=True,
            )
        )

    logger.info(f"Train for {msetup['epochs']} epochs ...")
    network.fit(
        train_generator,
        epochs=msetup["epochs"],
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=2,
        shuffle=False,
    )

    return network


def attention_train(data_folder: Path, train_folder: Path, setup: dict):
    """
    Attention Network training.

    Parameters
    ----------
        - data_folder: the input data folder path
        - train_folder: the train output folder path
        - setup: settings dictionary
    """
    # data loading
    train_generator, val_generator, test_generator = read_data(data_folder, setup)

    # model loading
    tfK.clear_session()
    msetup = setup["model"]["attention"]
    network = load_and_compile_network(msetup, setup["run_tf_eagerly"])
    network.summary()

    # training
    train_network(msetup, train_folder, network, (train_generator, val_generator))

    # inference
    network.evaluate(test_generator)

    res = network.predict(test_generator)
    tgts = test_generator.targets.astype(bool)

    import numpy as np
    import matplotlib.pyplot as plt

    scores_true = res[tgts]
    scores_false = res[~tgts]

    bins = np.linspace(0, 1, 101)
    h_true, _ = np.histogram(scores_true, bins=bins)
    h_false, _ = np.histogram(scores_false, bins=bins)
    plt.hist(bins[:-1], bins, weights=h_true, histtype="step", lw=0.5, color="red")
    plt.hist(bins[:-1], bins, weights=h_false, histtype="step", lw=0.5, color="green")
    plt.show()
