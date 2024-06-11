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
qsvm_connections = [[18,14,0,1], 
                    [2,3,4,5], 
                    [6, 7, 8,9], 
                    [10, 11, 12, 13], 
                    [19,20,21,22], 
                    [23, 24, 25, 26], 
                    [27, 28,29,30], 
                    [31, 32, 36, 51],
                    [37, 38, 39, 33], 
                    [40, 41,42, 53],#[40, 41, 42, 43],
                    [64, 54, 45, 46],
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
                    [108, 112, 116,125], #[107, 108, 112, 126],
                    [109, 114, 115, 116],
                    [117, 118, 119, 120],
                    [121, 122, 123, 124]]#[122, 123, 124, 125]]


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
        
        # ("single_non_parametric", ["I", "H", "X", "SX"]),
        # ("single_parametric", ["RX", "RY", "RZ"]),
        # ("two_non_parametric", ["CX"]),
        # ("two_parametric", ["CRX", "CRY", "CRZ"]),
    ]
)
coupling_map = None  # [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7]]
basis_gates = ['ecr', 'id', 'rz', 'sx', 'x']

generation_zero_old = genetic.initial_population(
    NB_INIT_INDIVIDUALS, NB_QUBITS, GATES_PER_QUBITS, gate_dict, nb_features
)
# CHANGE THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CHANGE THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CHANGE THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import pandas as pd
suffix = '2024_02_24 -13_49_58'
fitnesses = np.loadtxt("../../Output_genetic/"+suffix +
                       "/fitness_values_iter_"+suffix+".txt")
genes_0 = pd.read_csv("../../Output_genetic/"+suffix+"/genes" +
                 suffix+".csv", header=None, index_col=False).to_numpy()
generation_zero = np.array(genes_0[-27:])

# n_epochs = 34
# pop_size = 27
# keep_elitism = 3
# grouped_fitness = np.zeros((n_epochs + 1, pop_size))
# grouped_genes = np.zeros((n_epochs + 1, pop_size, 216))
# grouped_fitness[0] = fitnesses[:pop_size]
# grouped_genes[0, :, :] = genes_0[:pop_size, :]

# for i in range(n_epochs):
#     counter = (pop_size - keep_elitism)*i
#     best_old_idxs = np.argsort(grouped_fitness[i])[-keep_elitism:]
#     grouped_fitness[i+1] = np.concatenate([grouped_fitness[i][best_old_idxs],
#                                           fitnesses[pop_size + counter: 2*pop_size + counter - keep_elitism]])
#     grouped_genes[i+1, :] = np.concatenate([grouped_genes[i, :][best_old_idxs],
#                                           genes_0[pop_size + counter: 2*pop_size + counter - keep_elitism]])
# generation_zero = np.array(grouped_genes[-1], dtype=int)    
##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
    ##########################################################################################################
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
    fitness_score = accuracy + 0.25*density
    return fitness_score

############ look here
timestr = "2024_02_24 -13_49_58" # so next gen is n=5
save_path = "../../Output_genetic/" + timestr
Path(save_path).mkdir(exist_ok=True)

# genetic.gen0_qpu_run(
#     data_cv,
#     data_labels,
#     backend,
#     gate_dict,
#     nb_features,
#     GATES_PER_QUBITS,
#     NB_QUBITS,
#     False,
#     coupling_map,
#     basis_gates,
#     timestr,
#     fitness_function,
#     init_solutions=generation_zero,
#     qsvm_connections = qsvm_connections
# )

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
    "mutation_percent_genes": 3,
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
