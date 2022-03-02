""" This module contains utility functions of general interest. """
import logging
from typing import Union
import shutil
import yaml
from pathlib import Path, PosixPath
from quake import PACKAGE

logger = logging.getLogger(PACKAGE)


def path_constructor(loader, node):
    value = loader.construct_scalar(node)
    return Path(value)


def load_runcard(runcard_file: Union[Path,str]) -> dict:
    """
    Load runcard from yaml file.

    Parameters
    ----------
        - runcard_file: Path or str, the yaml to dump the dictionary
    
    Returns
    -------
        - dict: the loaded settings dictionary
    
    Note
    ----
    The pathlib.Path objects are automatically loaded if they are encoded
    with the following syntax:

    ```
    path: !Path 'path/to/file'
    ```
    """
    yaml.add_constructor('!Path', path_constructor)
    with open(runcard_file, "r") as stream:
        runcard = yaml.load(stream, Loader=yaml.FullLoader)
    return runcard


def path_representer(dumper, data):
    return dumper.represent_scalar('!Path', '%s' % data)


def save_runcard(fname: Union[Path,str], setup: dict):
    """
    Save runcard to yaml file.

    Parameters
    ----------
        - fname: Path or str, the yaml output file
        - setup: dict, the settings dictionary to be dumped
    
    Note
    ----
    pathlib.PosixPath objects are automatically loaded.    
    """
    yaml.add_representer(PosixPath, path_representer)
    with open(fname, "w") as f:
        yaml.dump(setup, f, indent=4)


def initialize_output_folder(output: Union[Path,str], should_force: bool):
    """
    Creates the output directory structure.

    Parameters
    ----------
        - output: Path, the ouptut directory
        - should_force: bool, wether to replace the already existing output directory
    """
    try:
        output.mkdir()
    except Exception as error:
        if should_force:
            logger.warning(f"Overwriting {output} with new model")
            shutil.rmtree(output)
            output.mkdir()
        else:
            msg = error + '\nDelete or run with "--force" to overwrite.'
            logger.error(msg)
            exit(-1)
    output.joinpath("cards").mkdir()
    output.joinpath("data").mkdir()

