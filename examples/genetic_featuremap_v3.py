""" This module executes an automatized quantum featuremap optimization via genetic algorithm using integer genes"""

from quake.utils.utils import load_runcard
import numpy as np
from pathlib import Path
from quake.models.qsvm.qsvm_tester import get_features
from collections import OrderedDict
from quake.models.qsvm import genetic_v3 as genetic
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Callable, Union

from qiskit import Aer

import time
import shutil
import os

# Dataset loading
data_folder = Path("../../output_2/data")
train_folder = Path("../../output_2/models/autoencoder")
setup = load_runcard("../../output_2/cards/runcard.yaml")
setup["run_tf_eagerly"] = True
setup["seed"] = 42

dataset, labels = get_features(data_folder.parent, "autoencoder", setup)
scaler = MinMaxScaler((0, 1)).fit(dataset[0])

data_cv, data_labels = genetic.get_subsample(dataset[2], labels[2], 50, scaler=scaler)
nb_features = data_cv.shape[1]

###########################################
############ Genetic settings #############
###########################################

NB_QUBITS = 6
GATES_PER_QUBITS = 5
NB_INIT_INDIVIDUALS = 30
gate_dict = OrderedDict(
    [
        ("single_non_parametric", ["I", "H", "X", "SX"]),
        ("single_parametric", ["RX", "RY", "RZ"]),
        ("two_non_parametric", ["CX"]),
        ("two_parametric", ["CRX", "CRY", "CRZ"]),
    ]
)
coupling_map = None  # [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7]]
basis_gates = None  # ['cx', 'id', 'rz', 'sx', 'x']

generation_zero = genetic.initial_population(
    NB_INIT_INDIVIDUALS, NB_QUBITS, GATES_PER_QUBITS, gate_dict, nb_features
)

gene_space = genetic.get_gene_space(gate_dict, nb_features, NB_QUBITS, GATES_PER_QUBITS)

# Defining the fitness function
def fitness_function(accuracy: float, density: float, depth: int) -> Union[np.float64, list[np.float64]]:
    """Customizable fitness function of some QSVM metrics.

    Parameters
    ----------
    accuracy: float
        5-fold cross validation accuracy.
    density: float
        Function of the off-diagonal kernel elements.
    depth: int
        Transpiled quantum circuit depth.

    Returns
    -------
    fitness_score: Union[np.float64, list[np.float64]]
        Quantum kernel fitness value. If it is a list, the run will be optimized with the NSGA-II algorithm for multi-objective optimization.
    """
    fitness_score = accuracy + 0.1*density
    return fitness_score


# Defining inputs for the genetic instance
options = {
    "num_generations": 50,
    "num_parents_mating": 15,
    "initial_population": generation_zero,
    "parent_selection_type": "rank",
    "mutation_by_replacement": True,
    "stop_criteria": "saturate_100",
    "mutation_type": "random",
    "mutation_percent_genes": 5,
    "crossover_probability": 0.2,
    "crossover_type": "two_points",
    "allow_duplicate_genes": True,
    "keep_elitism": 4,
    "fit_fun": fitness_function,
}

# Running the instance and retrieving data

backend = Aer.get_backend("statevector_simulator")
projected = False
timestr = time.strftime("%Y_%m_%d -%H_%M_%S")

ga_instance = genetic.genetic_instance(
    gene_space,
    data_cv,
    data_labels,
    backend,
    gate_dict,
    nb_features,
    GATES_PER_QUBITS,
    NB_QUBITS,
    projected,
    timestr,
    coupling_map=coupling_map,
    basis_gates=basis_gates,
    **options,
)

ga_instance.run()

solution, solution_fitness, _ = ga_instance.best_solution(
    ga_instance.last_generation_fitness
)
save_path = "../../Output_genetic/" + timestr
Path(save_path).mkdir(exist_ok=True)
with open(save_path + "/best_solution.txt", "w") as genes_file:
    np.savetxt(genes_file, solution)
with open(save_path + "/best_fitness.txt", "w") as file:
    file.write(str(solution_fitness) + "\n")
with open(save_path + "/best_fitness_per_generation.txt", "w") as file:
    file.write(str(ga_instance.best_solutions_fitness) + "\n")
np.save(save_path + "/data_cv", data_cv)
np.save(save_path + "/labels_cv", data_labels)

copied_script_name = time.strftime("%Y-%m-%d_%H%M") + "_" + os.path.basename(__file__)
shutil.copy(__file__, save_path + os.sep + copied_script_name)
