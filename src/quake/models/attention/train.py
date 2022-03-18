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


def load_and_compile_network(msetup: dict, run_tf_eagerly: bool) -> AttentionNetwork:
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
    if msetup["optimizer"] == "Adam":
        opt = Adam(learning_rate=lr)
    elif msetup["optimizer"] == "SGD":
        opt = SGD(learning_rate=lr)
    elif msetup["optimizer"] == "RMSprop":
        opt = RMSprop(learning_rate=lr)
    elif msetup["optimizer"] == "Adagrad":
        opt = Adagrad(learning_rate=lr)

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
            # histogram_freq=1,
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

    # evaluation
    network.evaluate(test_generator)