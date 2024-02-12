""" This module executes an automatized quantum featuremap optimization via genetic algorithm using integer genes on a QPU.
All the individuals in a generations are computed simultaneously in the same QPU by partitioning it into smaller computational units."""

from quake.utils.utils import load_runcard
import numpy as np
from pathlib import Path
from qiskit_ibm_runtime import QiskitRuntimeService, Session

from quake.models.qsvm.qsvm_tester import get_features
from collections import OrderedDict
from quake.models.qsvm import genetic_v4 as genetic
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Callable, Union

from qiskit import Aer

import time
import shutil
import os

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_provider import IBMProvider
# Save your credentials on disk.
# IBMProvider.save_account(token=) ADD WITH SESSION

provider = IBMProvider(instance='ibm-q-cern/infn/qcnphepgw')
backend = provider.get_backend('ibm_nazca')

qsvm_connections = [[18,14,0,1], 
                    [2,3,4,5], 
                    [6, 7, 8,9], 
                    [10, 11, 12, 13], 
                    [19,20,21,22], 
                    [23, 24, 25, 26], 
                    [27, 28,29,30], 
                    [31, 32, 36, 51],
                    [37, 38, 39, 33], 
                    [40, 41, 42, 43],
                    [64, 53, 45, 46],
                    [35, 47, 48, 49],
                    [52, 56, 57,58],
                    [60, 61, 62, 63],
                    [85, 73, 66, 67],
                    [55, 68, 69, 70],
                    [75, 76, 77,71],
                    [78, 79, 91, 98],
                    [72, 81, 82, 83],
                    [87, 88, 89, 74],
                    [90, 94, 95, 96],
                    [99, 100, 101, 102],
                    [103, 104, 105, 106],
                    [107, 108, 112, 126],
                    [109, 114, 115, 116],
                    [117, 118, 119, 120],
                    [122, 123, 124, 125]]


# backend = Aer.get_backend('qasm_simulator')
# Dataset loading
data_folder = Path("../../output_2/data")
train_folder = Path("../../output_2/models/autoencoder")
setup = load_runcard("../../output_2/cards/runcard.yaml")
setup["run_tf_eagerly"] = True
setup["seed"] = 42

dataset, labels = get_features(data_folder.parent, "autoencoder", setup)
scaler = MinMaxScaler((0, 1)).fit(dataset[0])

data_cv, data_labels = genetic.get_subsample(dataset[2], labels[2], 26, scaler=scaler)
nb_features = data_cv.shape[1]

###########################################
############ Genetic settings #############
###########################################
projected = False
timestr = time.strftime("%Y_%m_%d -%H_%M_%S")

NB_QUBITS = 4
GATES_PER_QUBITS = 3
NB_INIT_INDIVIDUALS = len(qsvm_connections)
gate_dict = OrderedDict(
    [
        ("single_non_parametric", ["I", "X", "SX"]),
        ("single_parametric", ["RZ"]),
        ("two_non_parametric", ["ECR"]),
        ("two_parametric", []),
        
        # ("single_non_parametric", ["I", "H", "X", "SX"]),
        # ("single_parametric", ["RX", "RY", "RZ"]),
        # ("two_non_parametric", ["CX"]),
        # ("two_parametric", ["CRX", "CRY", "CRZ"]),
    ]
)
coupling_map = None  # [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7]]
basis_gates = ['ecr', 'id', 'rz', 'sx', 'x']

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
    fitness_score = accuracy + 0.5*density
    return fitness_score

save_path = "../../Output_genetic/" + timestr
Path(save_path).mkdir(exist_ok=True)

genetic.gen0_qpu_run(
    data_cv,
    data_labels,
    backend,
    gate_dict,
    nb_features,
    GATES_PER_QUBITS,
    NB_QUBITS,
    False,
    coupling_map,
    basis_gates,
    timestr,
    fitness_function,
    init_solutions=generation_zero,
    qsvm_connections = qsvm_connections
)

# Defining inputs for the genetic instance
options = {
    "num_generations": 10,
    "num_parents_mating": 3,
    "initial_population": generation_zero,
    "parent_selection_type": "rank",
    "mutation_by_replacement": True,
    "stop_criteria": "saturate_100",
    "mutation_type": "random",
    "mutation_percent_genes": 10,
    "crossover_probability": 0.2,
    "crossover_type": "two_points",
    "allow_duplicate_genes": True,
    "keep_elitism": 1,
    "fit_fun": fitness_function,
    "qsvm_connections": qsvm_connections
}

# Running the instance and retrieving data

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
