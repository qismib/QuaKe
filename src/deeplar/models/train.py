"""
    This module calls the main training loops functions: reads the runcard in
    `<output folder>` and trains the query model on the stored data.

    Usage example:

    ```
    deeplar train --output <output folder> --model <modeltype>
    ```

"""
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from deeplar import PACKAGE
from deeplar.utils.utils import load_runcard, check_in_folder
from deeplar.utils.ask_edit_card import ask_edit_card

logger = logging.getLogger(PACKAGE + ".train")


def add_arguments_train(parser: ArgumentParser):
    """Adds train subparser arguments.

    Parameters
    ----------
    parser: ArgumentParser
        Train subparser object.
    """
    valid_models = ["cnn", "attention", "blob"]
    parser.add_argument("--output", "-o", type=Path, help="the output folder")
    parser.add_argument(
        "--model", "-m", type=str, help="the model to train", choices=valid_models
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="triggers interactive mode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random generator seed for reproducibility",
    )
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing files if present"
    )
    parser.add_argument("--debug", action="store_true", help="run tf in eager mode")
    parser.set_defaults(func=train)


def train(args: Namespace):
    """Train wrapper function: calls the train main function.

    Parameters
    ----------
    args: Namespace
        Command line parsed arguments..
    """
    # load runcard and setup output folder structure
    if args.interactive:
        ask_edit_card(logger, args.output)
    setup = load_runcard(args.output / "cards/runcard.yaml")
    setup.update({"seed": args.seed, "run_tf_eagerly": args.debug})

    check_in_folder(args.output / f"models/{args.model}", args.force)

    # launch main datagen function
    train_main(
        setup["output"] / "data",
        setup["output"] / f"models/{args.model}",
        args.model,
        setup,
    )


def preconfig_tf(setup: dict):
    """
    Set the host device for tensorflow.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus):
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        gpus = setup.get("gpu")
        if gpus:
            if isinstance(gpus, int):
                gpus = [gpus]
            gpus = [
                tf.config.PhysicalDevice(f"/physical_device:GPU:{gpu}", "GPU")
                for gpu in gpus
            ]
            tf.config.set_visible_devices(gpus, "GPU")
            logger.warning(f"Host device: GPU {gpus}")
        else:
            tf.config.set_visible_devices([], "GPU")
            logger.warning("Host device: CPU")
    else:
        logger.warning("Host device: CPU")

    if setup.get("debug"):
        logger.warning("Run all tf functions eagerly")
        tf.config.run_functions_eagerly(True)

    from deeplar.utils.configflow import set_manual_seed_tf

    set_manual_seed_tf(setup.get("seed"))


def train_main(data_folder: Path, train_folder: Path, modeltype: str, setup: dict):
    """Training main function that triggers model's training.

    The specific training function must implement the following steps:

    - data loading
    - model loading
    - training
    - evaluation

    Parameters
    ----------
    data_folder: Path
        The input data folder path.
    train_folder: Path
        The train output folder path.
    modeltype: str
        Available options cnn | attention.
    setup: dict
        Settings dictionary.

    Raises
    ------
    NotImplementedError
        If `modeltype` is not one of blob, cnn, attention
    """
    preconfig_tf(setup)
    from .attention.train import attention_train
    from .cnn.train import cnn_train
    from .blob.train import blob_train

    if modeltype == "cnn":
        logger.info("Training Convolutional Neural Network")
        cnn_train(data_folder, train_folder, setup)
    elif modeltype == "attention":
        logger.info("Training Attention Network")
        attention_train(data_folder, train_folder, setup)
    elif modeltype == "blob":
        logger.info("Classifying with Blob-method")
        blob_train(data_folder, train_folder, setup)
    else:
        raise NotImplementedError(f"model not implemented, found: {modeltype}")
