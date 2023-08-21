""" This model executes an automatized quantum featuremap optimization via genetic algorithm using integer genes"""

from quake.utils.utils import load_runcard, save_runcard
from quake.models.autoencoder.autoencoder_dataloading import read_data
from quake.models.autoencoder.train import load_and_compile_network
import numpy as np
from pathlib import Path
from quake.models.qsvm.qsvm_tester import get_features

from quake.models.qsvm import genetic_main as genetic

# Dataset loading
data_folder = Path("../../output_2/data")
train_folder = Path("../../output_2/models/autoencoder")
setup = load_runcard("../../output_2/cards/runcard.yaml")
setup["run_tf_eagerly"] = False
setup["seed"] = 42

dataset, labels = get_features(data_folder.parent, "autoencoder", setup)
data_train, lab_train = genetic.get_subsample(dataset[0], labels[0], 300)
data_val, lab_val = genetic.get_subsample(dataset[1], labels[1], 300)
nb_features = data_val.shape[1]

NB_QUBITS = 4
GATES_PER_QUBITS = 6
NB_INIT_INDIVIDUALS = 8
gate_dict = {"single_non_parametric": {"I", "H", "X", "S"},
             "single_parametric": {"RX", "RY", "RZ"},
             "two_non_parametric": {"CX"},
             "two_parametric": {"CRX", "CRY", "CRZ", "CP"}
}

generation_zero = genetic.initial_population(NB_INIT_INDIVIDUALS, NB_QUBITS, GATES_PER_QUBITS, gate_dict, nb_features)

genetic.to_quantum(generation_zero, gate_dict)
# use gene_space
