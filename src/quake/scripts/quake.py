"""
    This script is the QuaKe package entry point. Parses the subcommands from
    command line and calls the appropriate function to run.

    Main help output:

    ```
    usage: quake [-h] {preprocess} ...

    quake

    positional arguments:
      {datagen}
        datagen  generate voxelized dataset from root files.

    optional arguments:
      -h, --help    show this help message and exit
    ```

"""
import logging
import argparse
from time import time as tm
from quake import PACKAGE
from quake.dataset.generate import add_arguments_datagen

logger = logging.getLogger(PACKAGE)


def main():
    """Defines the QuaKe main entry point."""
    parser = argparse.ArgumentParser(description="quake")

    subparsers = parser.add_subparsers()

    # preprocess dataset
    gen_msg = "generate voxelized dataset from root files."
    gen_subparser = subparsers.add_parser(
        "datagen", description=gen_msg, help=gen_msg.lower().split(":")[0]
    )
    add_arguments_datagen(gen_subparser)

    args = parser.parse_args()

    # execute parsed function
    start = tm()
    args.func(args)
    logger.info(f"Program done in {tm()-start} s")
