""" This module executes an automatized quantum featuremap optimization via genetic algorithm using integer genes on a QPU.
All the individuals in a generations are computed simultaneously in the same QPU by partitioning it into smaller computational units."""

import numpy as np
from pathlib import Path
from qiskit_ibm_runtime import QiskitRuntimeService, Session

from collections import OrderedDict
from quake.models.qsvm import genetic_v4 as genetic
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Callable, Union


import time
import shutil
import os
import gc
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_provider import IBMProvider
# Save your credentials on disk.
IBMProvider.save_account(token='', overwrite=True)
#ADD WITH SESSION    
provider = IBMProvider(instance='ibm-q-cern/infn/qcnphepgw')
backend = provider.get_backend('ibm_brisbane')
qsvm_connections = [
    [2,1,0,14],
    [4,15,22,21],
    [6,7,8,16],
    [10, 11,12,13],
    [20,33,39,40],
    [17,30,29,28],
    [32,36,51,50],
    [41,53,60,59],
    [43,44,45,46],
    [62,63,64,54],
    [57,58,71,77],
    [85,84,83,82],
    [87,88,89,74],
    [76,75,90,94],
    [100,99,98,97],
    [105,104,103,102],
    [108,107,106,93],
    [114,109,96,95],
    [116,117,118,110],
    [125,124,123,122]
]


# backend = Aer.get_backend('qasm_simulator')
# Dataset loading

data_cv, data_labels = np.load("C:/Users/pc/Desktop/work/data_cv.npy"), np.load("C:/Users/pc/Desktop/work/data_labels.npy")
nb_features = data_cv.shape[1]
gc.collect()
###########################################
############ Genetic settings #############
###########################################
projected = False
timestr = time.strftime("%Y_%m_%d -%H_%M_%S")

NB_QUBITS = 4
GATES_PER_QUBITS = 9
NB_INIT_INDIVIDUALS = len(qsvm_connections)
gate_dict = OrderedDict(
    [
        ("single_non_parametric", ["Id", "X", "SX"]),
        ("single_parametric", ["RZ"]),
        ("two_non_parametric", ["ECR"]),
        ("two_parametric", []),
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
    fitness_score = accuracy + 0.1*density # first two epochs are with 0.25
    return fitness_score

############ look here
import pandas as pd
# timestr = "2024_04_09 -15_28_06"
save_path = "../../Output_genetic/" + timestr
Path(save_path).mkdir(exist_ok=True)
# genes_0 = pd.read_csv("../../Output_genetic/"+timestr+"/genes" +
#                  timestr+".csv", header=None, index_col=False).to_numpy()
# generation_zero = np.array(genes_0[-22:])

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
    "num_generations": 50,
    "num_parents_mating": 10,
    "keep_parents": -1,
    "initial_population": generation_zero,
    "parent_selection_type": "rank",
    "mutation_by_replacement": True,
    "stop_criteria": "saturate_100",
    "mutation_type": "random",
    "mutation_percent_genes": 5,
    "crossover_probability": 0.1,
    "crossover_type": "two_points",
    "allow_duplicate_genes": True,
    "keep_elitism": 3,
    "fit_fun": fitness_function,
    "qsvm_connections": qsvm_connections,
    "recovery_mode": False, ############ CHANGE WHEN NOT NEEDED
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
