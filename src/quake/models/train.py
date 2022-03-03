"""
    This module calls the main training loops functions: reads the runcard in
    `<output folder>` and trains the query model on the stored data.

    Usage example:

    ```
    quake train <output folder> --model <modeltype>
    ```

"""
import logging
from pathlib import Path
from quake import PACKAGE
from quake.utils.utils import load_runcard, check_in_folder
from .attention.train import attention_train

logger = logging.getLogger(PACKAGE + ".train")


def add_arguments_train(parser):
    """
    Adds train subparser arguments.

    Parameters
    ----------
        - parser: ArgumentParser, train subparser object
    """
    valid_models = ["svm", "cnn", "attention"]
    parser.add_argument("--output", "-o", type=Path, help="the output folder")
    parser.add_argument(
        "--model", "-m", type=str, help="the model to train", choices=valid_models
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


def train(args):
    """
    Train wrapper function: calls the train main function.

    Parameters
    ----------
        - args: NameSpace object, command line parsed arguments.
    """
    # load runcard and setup output folder structure
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


def train_main(data_folder: Path, train_folder: Path, modeltype: str, setup: dict):
    """
    Training main function that triggers model's training.
    The specific training function must implement the following steps:
        - data loading
        - model loading
        - training
        - evaluation

    Parameters
    ----------
        - data_folder: Path, the input data folder path
        - train_folder: Path, the train output folder path
        - modeltype: str, the model. Available options attention
        - setup: dict, settings dictionary
    """
    if modeltype == "svm":
        logger.info("Training SVM")
        logger.info("SVM not implemented yet. Exiting ...")
    elif modeltype == "cnn":
        logger.info("Training CNN")
        logger.info("CNN not implemented yet. Exiting ...")
    elif modeltype == "attention":
        logger.info("Training Attention Network")
        attention_train(data_folder, train_folder, setup)
    else:
        raise NotImplementedError(f"model not implemented, found: {modeltype}")
