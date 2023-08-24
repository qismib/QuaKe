""" This model executes an automatized quantum featuremap optimization via genetic algorithm using integer genes"""

from quake.utils.utils import load_runcard, save_runcard
from quake.models.autoencoder.autoencoder_dataloading import read_data
from quake.models.autoencoder.train import load_and_compile_network
import numpy as np
from pathlib import Path
from quake.models.qsvm.qsvm_tester import get_features

from quake.models.qsvm import genetic_main as genetic

from qiskit import Aer

import time

# Dataset loading
data_folder = Path("../../output_2/data")
train_folder = Path("../../output_2/models/autoencoder")
setup = load_runcard("../../output_2/cards/runcard.yaml")
setup["run_tf_eagerly"] = False
setup["seed"] = 42

dataset, labels = get_features(data_folder.parent, "autoencoder", setup)
data_cv, data_labels = genetic.get_subsample(dataset[2], labels[2], 100)
nb_features = data_cv.shape[1]

NB_QUBITS = 4
GATES_PER_QUBITS = 6
NB_INIT_INDIVIDUALS = 20
gate_dict = {
    "single_non_parametric": {"I", "H", "X", "S"},
    "single_parametric": {"RX", "RY", "RZ"},
    "two_non_parametric": {"CX"},
    "two_parametric": {"CRX", "CRY", "CRZ", "CP"},
}

generation_zero = genetic.initial_population(
    NB_INIT_INDIVIDUALS, NB_QUBITS, GATES_PER_QUBITS, gate_dict, nb_features
)
gene_space = genetic.get_gene_space(gate_dict, nb_features, NB_QUBITS, GATES_PER_QUBITS)


# Defining inputs for the genetic instance
options = {
    "num_generations": 100,
    "num_parents_mating": 5,
    "initial_population": generation_zero,
    "parent_selection_type": "rank",
    "mutation_by_replacement": True,
    "stop_criteria": "saturate_20",
    "mutation_type": "adaptive",
    "mutation_probability": [0.4, 0.15],
    "crossover_probability": 0.2,
    "crossover_type": "two_points",
}

# Running the instance and retrieving data

backend = Aer.get_backend("statevector_simulator")

timestr = time.strftime("%Y_%m_%d")  # -%H_%M_%S")

ga_instance = genetic.genetic_instance(
    options,
    gene_space,
    data_cv,
    data_labels,
    backend,
    gate_dict,
    nb_features,
    GATES_PER_QUBITS,
    NB_QUBITS,
    timestr,
)

ga_instance.run()
