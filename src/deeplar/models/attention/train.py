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
from ..AbstractNet import FeatureReturner
from deeplar import PACKAGE
from deeplar.utils.callbacks import DebuggingCallback
from deeplar.utils.diagnostics import (
    save_histogram_activations_image,
    save_scatterplot_features_image,
)

logger = logging.getLogger(PACKAGE + ".attention")


def load_and_compile_network(
    msetup: dict, run_tf_eagerly: bool, **kwargs
) -> AttentionNetwork:
    """Loads and compiles attention network.

    Parameters
    ----------
    msetup: dict
        Attention model settings dictionary.
    run_tf_eagerly: bool
        Wether to run tf eagerly, for debugging purposes.

    Returns
    -------
    network: AttentionNetwork
        The compiled network.
    """
    lr = float(msetup["lr"])
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
    """Trains the network.

    Parameters
    ----------
    msetup: dict
        Attention model settings dictionary.
    output: path
        The output folder.
    network: AttentionNetwork
        The network to be trained.
    generators: Tuple[Dataset, Dataset]
        The train and validation generators.

    Returns
    -------
    network: AttentionNetwork
        The trained network.
    """
    train_generator, val_generator = generators

    logdir = output / f"logs/{tm()}"
    checkpoint_filepath = output.joinpath("attention.h5").as_posix()
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
        DebuggingCallback(logdir=logdir / "validation", validation_data=val_generator),
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


def make_inference_plots(
    train_folder: Path,
    network: AttentionNetwork,
    test_generator: tf.keras.utils.Sequence,
):
    """Plots accuracy plots.

    Parameters
    ----------
    train_folder: Path
        The train output folder path.
    network: AttentionNetwork
        The trained network.
    test_generator: tf.keras.utils.Sequence
        Test generator
    """
    with FeatureReturner(network) as fr:
        y_pred, features = fr.predict(test_generator, verbose=1)
    y_true = test_generator.targets

    fname = train_folder / "scatterplot_features.svg"
    save_scatterplot_features_image(fname, features, y_true)

    fname = train_folder / "histogram_scores.svg"
    save_histogram_activations_image(fname, y_pred, y_true)


def attention_train(data_folder: Path, train_folder: Path, setup: dict):
    """Attention Network training.

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
    msetup = setup["model"]["attention"]
    network = load_and_compile_network(msetup, setup["run_tf_eagerly"])
    network.summary()
    tf.keras.utils.plot_model(
        network.model(),
        to_file=train_folder / "attention_network.png",
        expand_nested=True,
        show_shapes=True,
    )
    # exit()
    # training
    train_network(msetup, train_folder, network, (train_generator, val_generator))

    # inference
    msetup.update({"ckpt": train_folder.parent / f"attention/attention.h5"})
    network = load_and_compile_network(msetup, setup["run_tf_eagerly"])
    network.evaluate(test_generator)

    make_inference_plots(train_folder, network, test_generator)

    # Delete this
    results = network.evaluate(test_generator)
    with open("/home/rmoretti/TESI/output_perf_attention/test/accuracy.txt", "a") as f:
        f.write(str(results))
        f.write("\n")
        f.write(str(setup["detector"]["min_energy"]))
        f.write("\n")

