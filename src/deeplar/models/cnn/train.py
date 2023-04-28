""" This module provides functions for CNN network training and loading."""
import logging
import math
import numpy as np
from typing import Tuple
from time import time as tm
from pathlib import Path
import tensorflow as tf
import tensorflow.keras.backend as tfK
from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from .cnn_dataloading import read_data, Dataset
from .cnn_network import CNN_Network
from deeplar import PACKAGE
from deeplar.dataset.generate_utils import Geometry
from deeplar.models.attention.train import make_inference_plots

# from deeplar.utils.callbacks import DebuggingCallback

logger = logging.getLogger(PACKAGE + ".cnn")


def load_and_compile_network(
    msetup: dict, run_tf_eagerly: bool = False, geo: Geometry = None
) -> CNN_Network:
    """Loads and compiles attention network.

    Parameters
    ----------
    msetup: dict
        CNN model settings dictionary.
    run_tf_eagerly: bool
        Wether to run tf eagerly, for debugging purposes.
    geo: Geometry
        Object describing detector geometry.

    Returns
    -------
    network: CNN_Network
        The compiled network.
    """
    lr = float(msetup["lr"])
    opt_kwarg = {
        # "clipvalue": 0.5,
    }
    if msetup["optimizer"].lower() == "adam":
        opt = Adam(learning_rate=lr, **opt_kwarg)
    elif msetup["optimizer"].lower() == "sgd":
        opt = SGD(learning_rate=lr, **opt_kwarg)
    elif msetup["optimizer"].lower() == "rmsprop":
        opt = RMSprop(learning_rate=lr, **opt_kwarg)
    elif msetup["optimizer"].lower() == "adagrad":
        opt = Adagrad(learning_rate=lr, **opt_kwarg)

    if hasattr(geo, "nb_xbins_reduced"):
        bins_number = [geo.nb_xbins_reduced, geo.nb_ybins_reduced, geo.nb_zbins_reduced]
    else:
        bins_number = [geo.nb_xbins, geo.nb_ybins, geo.nb_zbins]

    network = CNN_Network(nb_bins=bins_number, **msetup["net_dict"])
    loss = tf.keras.losses.BinaryCrossentropy(name="xent")
    # loss = tf.keras.losses.LogCosh(name = "xent")
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="acc"),
        tf.keras.metrics.Precision(name="prec"),
        tf.keras.metrics.Recall(name="rec"),
        tf.keras.metrics.AUC(name="auc"),
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
    network: CNN_Network,
    generators: Tuple[Dataset, Dataset],
) -> CNN_Network:
    """Trains the CNN network.

    Parameters
    ----------
    msetup: dict
        Attention model settings dictionary.
    output: Path
        The output folder.
    network:CNN_Network
        The network to be trained.
    generators: Tuple[Dataset, Dataset]
        The train and validation generators.

    Returns
    -------
    network: CNN_Network:
        The trained network.
    """
    train_generator, val_generator = generators

    logdir = output / f"logs/{tm()}"
    checkpoint_filepath = output.joinpath("cnn.h5").as_posix()

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
            factor=0.5,
            mode="max",
            verbose=2,
            patience=msetup["reducelr_patience"],
            min_lr=float(
                msetup["min_lr"],
            ),
        ),
        TensorBoard(
            log_dir=logdir,
            # write_graph=True,
            # write_images=True,
            # update_freq='batch',
            # histogram_freq=5,
            # profile_batch=5,
        ),
        #  DebuggingCallback(logdir=logdir / "validation", validation_data=val_generator),
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


def cnn_train(data_folder: Path, train_folder: Path, setup: dict):
    """CNN Network training.

    Parameters
    ----------
    data_folder: Path
        The input data folder path.
    train_folder: Path
        The train output folder path.
    setup: dict
        Settings dictionary.
    """
    # data loading
    train_generator, val_generator, test_generator = read_data(
        data_folder, train_folder, setup
    )

    # model loading
    tfK.clear_session()
    msetup = setup["model"]["cnn"]
    geo = Geometry(setup["detector"])
    network = load_and_compile_network(msetup, setup["run_tf_eagerly"], geo=geo)
    network.summary()
    tf.keras.utils.plot_model(
        network.model(),
        to_file=train_folder / "cnn_network.png",
        expand_nested=True,
        show_shapes=True,
    )
    # training
    train_network(msetup, train_folder, network, (train_generator, val_generator))
    # inference
    msetup.update({"ckpt": train_folder.parent / f"cnn/cnn.h5"})
    network = load_and_compile_network(msetup, setup["run_tf_eagerly"], geo=geo)
    network.evaluate(test_generator)
    make_inference_plots(train_folder, network, test_generator)

    results = network.evaluate(test_generator)
    with open("/home/rmoretti/TESI/output_perf_cnn/test/accuracy.txt", "a") as f:
        f.write(str(results))
        f.write("\n")
        f.write(str(setup["detector"]["min_energy"]))
        f.write("\n")
