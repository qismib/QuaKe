"""
    This script is the QuaKe package entry point. Parses the subcommands from
    command line and calls the appropriate function to run.

    Main help output:

    ```
    usage: deeplar [-h] {datagen,train} ...
    
    deeplar
    
    positional arguments:
      {datagen,train}
        datagen        generate voxelized dataset from root files
        train          train model
    
    optional arguments:
      -h, --help       show this help message and exit
    
    ```

"""
import logging
import argparse
from time import time as tm
from deeplar import PACKAGE
from deeplar.dataset.generate import add_arguments_datagen
from deeplar.models.train import add_arguments_train

logger = logging.getLogger(PACKAGE)


def main():
    """Defines the QuaKe main entry point."""
    parser = argparse.ArgumentParser(description="deeplar")

    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    parser.add_argument(
        "-l", default=None, help="set logging level", choices=log_levels, dest="logging"
    )

    subparsers = parser.add_subparsers()

    # preprocess dataset
    gen_msg = "generate voxelized dataset from root files"
    gen_subparser = subparsers.add_parser(
        "datagen", description=gen_msg, help=gen_msg.lower().split(":")[0]
    )
    add_arguments_datagen(gen_subparser)

    # train
    train_msg = "train model"
    train_subparser = subparsers.add_parser(
        "train", description=train_msg, help=train_msg.lower().split(":")[0]
    )
    add_arguments_train(train_subparser)

    args = parser.parse_args()

    # setting global logging level
    if args.logging:
        logger.warning(f"Setting log level to {args.logging}")
        logger.setLevel(args.logging)
        logger.handlers[0].setLevel(args.logging)

    # execute parsed function
    start = tm()
    args.func(args)
    logger.info(f"Program done in {tm()-start} s")
