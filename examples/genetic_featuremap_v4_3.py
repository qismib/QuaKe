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
backend = provider.get_backend('ibm_nazca')

qsvm_connections = [
    [0,1,2,3],
    [4,15,22,21],
    [25,26,16,8],
    [13,12,11,10],
    [20, 33,39,38],
    [27,28,29,30],
    [32,36,51,50],
    [40,41,53,60],
    [49,48,47,46],
    # [71,58,57,56],
    [64,63,62,72],
    [70,69,68,67],
    [77,78,79,80],
    [82,83,84,85],
    [87,88,89,74],
    [95,94,90,75],
    [96,97,98,99],
    [119,118,110,100],
    [104,105,106,93],
    [116,115,114,113],
    [111,122,121,120],
    [126,125,124,123]
]


# backend = Aer.get_backend('qasm_simulator')
# Dataset loading

data_cv, data_labels = np.load("C:/Users/pc/Desktop/work/data_cv.npy"), np.load("C:/Users/pc/Desktop/work/data_labels.npy")
useful_ft = [0, 1,2,5,6, 8, 10, 12, 14, 15, 17]
data_cv = data_cv[:, useful_ft]
nb_features = data_cv.shape[1]
gc.collect()
###########################################
############ Genetic settings #############
###########################################
projected = False
timestr = time.strftime("%Y_%m_%d -%H_%M_%S")
recovery_mode = False
NB_QUBITS = 4
GATES_PER_QUBITS = 6
NB_INIT_INDIVIDUALS = len(qsvm_connections)
gate_dict = OrderedDict(
    [
        ("single_non_parametric", ["Id","Id", "X","X","SX","SX"]),
        ("single_parametric", ["RZ", "RZ"]),
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
    fitness_score = accuracy + 0.05*density # first two epochs are with 0.25
    return fitness_score

############ look here
import pandas as pd
# timestr = "2024_04_11 -23_51_52"
save_path = "../../Output_genetic/" + timestr
Path(save_path).mkdir(exist_ok=True)

# genes_0 = pd.read_csv("../../Output_genetic/"+timestr+"/genes" +
#                  timestr+".csv", header=None, index_col=False).to_numpy()
# generation_zero = np.array(genes_0[-21:])

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
    qsvm_connections = qsvm_connections,
    recovery_mode=recovery_mode,
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
    "recovery_mode": recovery_mode,  ############ CHANGE WHEN NOT NEEDED
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
