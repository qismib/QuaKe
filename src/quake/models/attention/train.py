import logging
from pathlib import Path
import tensorflow.keras.backend as tfK
from .attention_dataloading import read_data
from .load_attention_network import load_and_compile_network, train_network
from quake import PACKAGE

logger = logging.getLogger(PACKAGE + ".attention")


def attention_train(data_folder: Path, train_folder: Path, setup: dict):
    """
    Attention Network training.

    Parameters
    ----------
        - data_folder: Path, the input data folder path
        - train_folder: Path, the train output folder path
        - setup: dict, settings dictionary
    """
    # data loading
    train_generator, val_generator, test_generator = read_data(data_folder, setup)

    # model loading
    tfK.clear_session()
    msetup = setup["model"]["attention"]
    network = load_and_compile_network(msetup, setup["run_tf_eagerly"])
    network.summary()

    # training
    # train_network(msetup, train_folder, network, (train_gen, test_gen))
    train_network(msetup, train_folder, network, (train_generator, val_generator))

    # evaluation
    network.evaluate(test_generator)
