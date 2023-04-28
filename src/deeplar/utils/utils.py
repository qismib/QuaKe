""" This module contains utility functions of general interest. """
import logging
import shutil
import yaml
import random
import numpy as np

# import tensorflow as tf
from pathlib import Path, PosixPath
from deeplar import PACKAGE

logger = logging.getLogger(PACKAGE)


def path_constructor(loader, node):
    """PyYaml utility function."""
    value = loader.construct_scalar(node)
    return Path(value)


def load_runcard(runcard_file: Path) -> dict:
    """Load runcard from yaml file.
    Parameters
    ----------
    runcard_file: Path
        The yaml to dump the dictionary.
    Returns
    -------
    runcard: dict
        The loaded settings dictionary.
    Note
    ----
    The pathlib.Path objects are automatically loaded if they are encoded
    with the following syntax:
    ```
    path: !Path 'path/to/file'
    ```
    """
    if not isinstance(runcard_file, Path):
        runcard_file = Path(runcard_file)
    yaml.add_constructor("!Path", path_constructor)
    with open(runcard_file, "r") as stream:
        runcard = yaml.load(stream, Loader=yaml.FullLoader)
    return runcard


def path_representer(dumper, data):
    """PyYaml utility function."""
    return dumper.represent_scalar("!Path", "%s" % data)


def save_runcard(fname: Path, setup: dict):
    """Save runcard to yaml file.
    Parameters
    ----------
    fname: Path
        The yaml output file.
    setup: Path
        The settings dictionary to be dumped.
    Note
    ----
    pathlib.PosixPath objects are automatically loaded.
    """
    yaml.add_representer(PosixPath, path_representer)
    with open(fname, "w") as f:
        yaml.dump(setup, f, indent=4)


def check_in_folder(folder: Path, should_force: bool):
    """Creates the query folder.
    The `should_force` parameters controls the function behavior in case
    `folder` exists. If true, it overwrites the existent directory, otherwise
    exit.
    Parameters
    ----------
    folder: Path
        The directory to be checked.
    should_force: bool
        Wether to replace the already existing directory.
    """
    try:
        folder.mkdir()
    except Exception as error:
        if should_force:
            logger.warning(f"Overwriting {folder} with new model")
            shutil.rmtree(folder)
            folder.mkdir()
        else:
            logger.error(error)
            logger.error('Delete or run with "--force" to overwrite.')
            exit(-1)


def initialize_output_folder(output: Path, should_force: bool):
    """Creates the output directory structure.
    Parameters
    ----------
    output: Path
        The output directory.
    should_force: bool
        Wether to replace the already existing output directory.
    """
    check_in_folder(output, should_force)
    output.joinpath("cards").mkdir()
    output.joinpath("data").mkdir()
    output.joinpath("models").mkdir()


def set_manual_seed(seed: int):
    """Set random and numpy libraries random seed for reproducibility.
    Parameters
    ----------
    seed: int
        Random generator seed for code reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
