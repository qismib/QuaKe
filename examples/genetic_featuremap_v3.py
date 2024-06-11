""" This module executes an automatized quantum featuremap optimization via genetic algorithm using integer genes"""

# from quake.utils.utils import load_runcard
import numpy as np
from pathlib import Path
# from quake.models.qsvm.qsvm_tester import get_features
from collections import OrderedDict
from quake.models.qsvm import genetic_v3 as genetic
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Callable, Union

from qiskit_aer import StatevectorSimulator

import time
import shutil
import os

# Dataset loading
# data_folder = Path("../../output_2/data")
# train_folder = Path("../../output_2/models/autoencoder")
# setup = load_runcard("../../output_2/cards/runcard.yaml")
# setup["run_tf_eagerly"] = True
# setup["seed"] = 42

# dataset, labels = get_features(data_folder.parent, "autoencoder", setup)
# scaler = MinMaxScaler((0, 1)).fit(dataset[0])

# data_cv, data_labels = genetic.get_subsample(dataset[2], labels[2], 50, scaler=scaler)
# nb_features = data_cv.shape[1]
data_cv, data_labels = np.load("C:/Users/pc/Desktop/work/data_1000.npy"), np.load("C:/Users/pc/Desktop/work/labels_1000.npy")
data_cv = data_cv[:100]
data_labels = data_labels[:100]

useful_ft = [0, 5, 6, 8, 10, 12, 14, 15]
data_cv = data_cv[:, useful_ft]

nb_features = data_cv.shape[1]

# data_cv, data_labels = np.load("C:/Users/pc/Desktop/work/data_cv.npy"), np.load("C:/Users/pc/Desktop/work/data_labels.npy")
# nb_features = data_cv.shape[1]

###########################################
############ Genetic settings #############
###########################################

NB_QUBITS = 4
GATES_PER_QUBITS = 8
NB_INIT_INDIVIDUALS = 30
# gate_dict = OrderedDict(
#     [
#         ("single_non_parametric", ["Id", "H", "X", "SX"]),
#         ("single_parametric", ["RX", "RY", "RZ"]),
#         ("two_non_parametric", ["CX"]),
#         ("two_parametric", ["CRX", "CRY", "CRZ"]),
#     ]
# )

gate_dict = OrderedDict(
    [
        ("single_non_parametric", ["Id", "X","SX"]),
        ("single_parametric", ["RZ", "RZ"]),
        ("two_non_parametric", ["ECR"]),
        ("two_parametric", []),
    ]
)
# One could instead encode gate-block (involving more than one qubit for example) and minimize metrix like depth or optimize particular transpilings...
coupling_map = None # [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7]]
basis_gates = None # ['cx', 'id', 'rz', 'sx', 'x'] 

generation_zero = genetic.initial_population(
    NB_INIT_INDIVIDUALS, NB_QUBITS, GATES_PER_QUBITS, gate_dict, nb_features
)

gene_space = genetic.get_gene_space(gate_dict, nb_features, NB_QUBITS, GATES_PER_QUBITS)

# Defining the fitness function
# def fitness_function(fmap = None) -> Union[np.float64, list[np.float64]]:
#     # fitness_score = accuracy + 0.05*density
#     count_ops = fmap.count_ops()
#     if 'cx' in count_ops:
#         num_2qb_gates = count_ops['cx']
#     else:
#         num_2qb_gates = 0
#     fitness_score = num_2qb_gates -fmap.depth()/2
#     return fitness_score

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
    fitness_score = accuracy + 0.025*density
    return fitness_score

# Defining inputs for the genetic instance
options = {
    "num_generations": 2,
    "num_parents_mating": 20,
    "initial_population": generation_zero,
    "parent_selection_type": "sss", # tournament_nsga2 # rank works really bad
    "mutation_by_replacement": True,
    "stop_criteria": "saturate_250",
    "mutation_type": "random",
    "mutation_percent_genes": 1.5 * 4 / NB_QUBITS * 8 / GATES_PER_QUBITS,
    "crossover_probability": 0.1,
    "crossover_type": "two_points",
    "allow_duplicate_genes": True,
    "keep_elitism": 0,
    "fit_fun": fitness_function,
    "noise_std": 0.04408471934506117,
}

# Running the instance and retrieving data
backend = StatevectorSimulator(precision='single')
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
