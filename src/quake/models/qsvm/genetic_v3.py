""" This module contains functions for generating a genetic-optimized quantum featuremap. 
This module uses integer encoding instead of binary encoding and allows for more search options than 'genetic_v1.py' and 'genetic_v2.py'"""

import csv
import time
from pathlib import Path
from typing import Tuple, Callable, Union
import pygad
import numpy as np
import inspect

from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit.quantum_info import partial_trace, DensityMatrix
from qiskit.compiler import transpile
from qiskit_aer.backends.statevector_simulator import StatevectorSimulator

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

def get_subsample(
    dataset: np.ndarray,
    labels: np.ndarray,
    size: int,
    seed: int = 42,
    scaler: MinMaxScaler = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Getting a smaller subsample of a given dataset.

    Parameters
    ----------
    dataset: np.ndarray
        The features distribution.
    labels: np.ndarray
        The truth labels.
    size: int
        Size of the subsample.
    seed: int
        Seed initialization for reproducibility.
    scaler: MinMaxScaler
        Scikit-Learn dataset scaler class.

    Returns
    -------
    subs_dataset: np.ndarray
        Subsampling of the feature distribution.
    subs_labels: np.ndarray
        Subsample truth labels.
    """
    
    min_accepted_value, max_accepted_value = scaler.feature_range
    dataset = scaler.transform(dataset)
    is_outlier = np.sum(np.logical_or(dataset >= max_accepted_value, dataset <= min_accepted_value), axis=1)
    dataset = dataset[is_outlier == 0]
    labels = labels[is_outlier == 0]
    subs_dataset, subs_labels = train_test_split(
        dataset, labels, train_size=size, random_state=seed
    )[::2]
    return subs_dataset, subs_labels


def initial_population(
    nb_init_individuals: int,
    nb_qubits: int,
    gates_per_qubits: int,
    gate_dict: dict,
    nb_features: int,
) -> np.array:
    """Initializing a population of chromosomes (generation 0)

    Parameters
    ----------
    nb_init_individuals: int
        Number of chromosomes in the population.
    nb_qubits: int
        Number of qubits for the genetic run.
    gates_per_qubits:
        Number of gates generated per qubit.
    gate_dict: dict
        Dictionary containing the allowed gates.
    nb_features:
        Number of features in the dataset.

    Returns:
    ----------
    gene_array: np.array
        An array describing the initial population genetic pool.
    """
    nb_possible_gates = (
        len(gate_dict["single_non_parametric"])
        + len(gate_dict["single_parametric"])
        + len(gate_dict["two_non_parametric"])
        + len(gate_dict["two_parametric"])
    )

    size_per_gene = nb_qubits * gates_per_qubits * nb_init_individuals
    gate_idxs = gen_int(0, nb_possible_gates, size=size_per_gene)
    feature_transformation = gen_int(0, 3, size=size_per_gene)
    multi_features = gen_int(0, 2, size=size_per_gene)
    first_feature_idx = gen_int(0, nb_features, size=size_per_gene)
    second_feature_idx = gen_int(
        0, nb_features, size=size_per_gene, exclude_array=first_feature_idx
    )
    second_qubit_idx = gen_int(
        0,
        nb_qubits,
        size=size_per_gene,
        exclude_array=np.tile(
            np.arange(0, nb_qubits), gates_per_qubits * nb_init_individuals
        ),
    )
    gene_array = np.array(
        [
            gate_idxs,
            feature_transformation,
            multi_features,
            first_feature_idx,
            second_feature_idx,
            # second_qubit_idx,
        ]
    )
    gene_array = np.reshape(
        gene_array.T, [nb_init_individuals, gates_per_qubits, nb_qubits, 5]
    ).reshape(nb_init_individuals, -1)
    return gene_array


# def unflatten_gene_list(
#     gene_list_flat, nb_init_individuals, gates_per_qubits, nb_qubits
# ):
#     gene_list = np.reshape(
#         gene_list_flat, [nb_init_individuals, gates_per_qubits, nb_qubits, 6]
#     )
#     return gene_list


def get_gene_space(
    gate_dict: dict, nb_features: int, nb_qubits: int, gates_per_qubits: int
) -> list[int]:
    """Computing the gene_space (min and max values), which is different according to the gene meaning in a gene sequence that describe a gate.

    Parameters
    ----------
    gate_dict: dict
        Dictionary containing the allowed gates.
    nb_features: int
        Number of features in the dataset.
    nb_qubits: int
        Number of qubits for the genetic run.
    gates_per_qubits: int
        Number of gates generated per qubit.

    Returns:
    ----------
    gene_space: list[int]
        Span values for all the genes in a chromosome.
    """
    nb_possible_gates = (
        len(gate_dict["single_non_parametric"])
        + len(gate_dict["single_parametric"])
        + len(gate_dict["two_non_parametric"])
        + len(gate_dict["two_parametric"])
    )
    size_per_gene = nb_qubits * gates_per_qubits
    gene_space = []
    for _ in range(size_per_gene):
        gene_space = gene_space + [
            range(nb_possible_gates),
            range(2),
            range(2),
            range(nb_features),
            range(nb_features),
            # range(nb_qubits),
        ]
    return gene_space


def gen_int(
    min_val: int, max_val: int, size: int = None, exclude_array: np.ndarray = None
):
    """Generating an integer sequence in a specific range, optionally excluding unwanted values that can be different according to the position in the array.

    Parameters
    ----------
    min_val: int
        Lower boundary.
    max_val: int
        Upper boundary.
    size: int
        Array length.
    exclude_array: np.ndarray
        Array containing a specific value to exclude in range, for any generated number.

    Returns:
    ----------
    random_indices: np.ndarray
        Integer sequence generated with the correct size, boundaries and exclusion settings.
    """
    if exclude_array is not None:
        valid_values_per_index = [
            np.setdiff1d(np.arange(min_val, max_val), [exclude_array[i]])
            for i in range(size)
        ]
        random_indices = [
            np.random.choice(valid_values_per_index[i]) for i in range(size)
        ]
    else:
        random_indices = np.random.randint(min_val, max_val, size)
    return random_indices

# def to_quantum(
#     genes: np.ndarray,
#     gate_dict: dict,
#     nb_features: int,
#     gates_per_qubits: int,
#     nb_qubits: int,
# ) -> Tuple[QuantumCircuit, list[int]]:
#     """Converting genes from an integer sequence to a quantum featuremap.

#     Parameters
#     ----------
#     genes: np.ndarray
#         Gene array defining a chromosome.

#     gate_dict: dict
#         Dictionary containing the allowed gates.
#     nb_features: int
#         Number of features in the dataset.
#     gates_per_qubits: int
#         Number of gates generated per qubit.
#     nb_qubits: int
#         Number of qubits for the genetic run.
#     gates_per_qubits: int
#         Number of gates generated per qubit.

#     Returns:
#     ----------
#     fmap: QuantumCircuit
#         Quantum featuremap.
#     x_idxs:
#         Features indices used in the featuremap out of the total.
#     """
#     gate_list = []
#     for gate_set in gate_dict.values():
#         gate_list = gate_list + list(gate_set)

#     genes_unflatted = np.reshape(genes, [gates_per_qubits, nb_qubits, 6])
#     x = ParameterVector("x", length=nb_features)
#     fmap = QuantumCircuit(nb_qubits)
#     x_idxs = []
#     for j in range(gates_per_qubits):
#         for k in range(nb_qubits):
#             gate_type_idx = genes_unflatted[j, k, 0]
#             feature_transformation_type = genes_unflatted[j, k, 1]
#             multi_features = genes_unflatted[j, k, 2]
#             first_feature_idx = genes_unflatted[j, k, 3]
#             second_feature_idx = genes_unflatted[j, k, 4]
#             second_qubit_idx = genes_unflatted[j, k, 5]
#             if second_qubit_idx == k:
#                 second_qubit_idx = (second_qubit_idx + 1) % nb_qubits
#             gate = getattr(fmap, gate_list[gate_type_idx].lower())
#             if gate_list[gate_type_idx] in gate_dict["single_non_parametric"]:
#                 if gate_list[gate_type_idx] != "Id":
#                     gate(k)
#             elif gate_list[gate_type_idx] in gate_dict["two_non_parametric"]:
#                 gate(k, second_qubit_idx)
#             else:
#                 if first_feature_idx not in x_idxs:
#                     x_idxs.append(first_feature_idx)

#                 if multi_features == 0 and feature_transformation_type == 0:
#                     param_expression = 2 * np.pi * (x[first_feature_idx] - 0.5)
#                 if multi_features == 1 and feature_transformation_type == 0:
#                     if second_feature_idx not in x_idxs:
#                         x_idxs.append(second_feature_idx)
#                     param_expression = (
#                         2 * np.pi * x[first_feature_idx] * (1 - x[second_feature_idx])
#                         - np.pi
#                     )
#                 if multi_features == 0 and feature_transformation_type == 1:
#                     param_expression = (
#                         2 * np.pi * x[first_feature_idx] * (1 - x[first_feature_idx])
#                         - np.pi
#                     )
#                 if multi_features == 1 and feature_transformation_type == 1:
#                     if second_feature_idx not in x_idxs:
#                         x_idxs.append(second_feature_idx)
#                     param_expression = (
#                         (
#                             2
#                             * np.pi
#                             * x[first_feature_idx]
#                             * (1 - x[second_feature_idx])
#                             - np.pi
#                         )
#                         * (
#                             2
#                             * np.pi
#                             * x[second_feature_idx]
#                             * (1 - x[first_feature_idx])
#                             - np.pi
#                         )
#                         / np.pi
#                     )

#                 if multi_features == 0 and feature_transformation_type == 2:
#                     param_expression = (
#                         2 * np.arcsin(2 * x[first_feature_idx] - 1) - np.pi
#                     )
#                 if multi_features == 1 and feature_transformation_type == 2:
#                     if second_feature_idx not in x_idxs:
#                         x_idxs.append(second_feature_idx)
#                     param_expression = 2 * np.arcsin(
#                         (2 * x[first_feature_idx] - 1) * (2 * x[second_feature_idx] - 1)
#                     )

#                 if gate_list[gate_type_idx] in gate_dict["single_parametric"]:
#                     gate(param_expression, k)
#                 elif gate_list[gate_type_idx] in gate_dict["two_parametric"]:
#                     gate(param_expression, k, second_qubit_idx)
#     return fmap, x_idxs

def to_quantum(
    genes: np.ndarray,
    gate_dict: dict,
    nb_features: int,
    gates_per_qubits: int,
    nb_qubits: int,
) -> Tuple[QuantumCircuit, list[int]]:
    """Converting genes from an integer sequence to a quantum featuremap.

    Parameters
    ----------
    genes: np.ndarray
        Gene array defining a chromosome.

    gate_dict: dict
        Dictionary containing the allowed gates.
    nb_features: int
        Number of features in the dataset.
    gates_per_qubits: int
        Number of gates generated per qubit.
    nb_qubits: int
        Number of qubits for the genetic run.
    gates_per_qubits: int
        Number of gates generated per qubit.

    Returns:
    ----------
    fmap: QuantumCircuit
        Quantum featuremap.
    x_idxs:
        Features indices used in the featuremap out of the total.
    """
    physical_qubits = [0,1,2,3]
    gate_list = []
    for gate_set in gate_dict.values():
        gate_list = gate_list + list(gate_set)

    genes_unflatted = np.reshape(genes, [gates_per_qubits, nb_qubits, 5])
    x = ParameterVector("x", length=nb_features )
    fmap = QuantumCircuit(nb_qubits)
    x_idxs = []
    for j in range(gates_per_qubits):
        for k in range(nb_qubits):
            gate_type_idx = genes_unflatted[j, k, 0]
            feature_transformation_type = genes_unflatted[j, k, 1]
            multi_features = genes_unflatted[j, k, 2]
            first_feature_idx = genes_unflatted[j, k, 3]
            second_feature_idx = genes_unflatted[j, k, 4]
            # second_qubit_idx = genes_unflatted[j, k, 5]
            # if second_qubit_idx == k:
            #     second_qubit_idx = (second_qubit_idx + 1) % nb_qubits
            gate = getattr(fmap, gate_list[gate_type_idx].lower())
            if gate_list[gate_type_idx] in gate_dict["single_non_parametric"]:
                if gate_list[gate_type_idx] != "I":
                    gate(k)
            elif gate_list[gate_type_idx] in gate_dict["two_non_parametric"]:
                did_something = False
                if k == 0 or k == 1:
                    gate(k, k+1)
                    did_something = True                      
                elif k == 3:
                    gate(k, k-1)
                    did_something = True
                if not did_something:
                    print(physical_qubits, "for qubit", physical_qubits[k], "I did nothing")
            else:
                if first_feature_idx not in x_idxs:
                    x_idxs.append(first_feature_idx)

                if multi_features == 0 and feature_transformation_type == 0:
                    param_expression = 2 * np.pi * (x[first_feature_idx] - 0.5)
                if multi_features == 1 and feature_transformation_type == 0:
                    if second_feature_idx not in x_idxs:
                        x_idxs.append(second_feature_idx)
                    param_expression = (
                        2 * np.pi * x[first_feature_idx] *
                        (1 - x[second_feature_idx])
                        - np.pi
                    )
                if multi_features == 0 and feature_transformation_type == 1:
                    param_expression = (
                        2 * np.pi * x[first_feature_idx] *
                        (1 - x[first_feature_idx])
                        - np.pi
                    )
                if multi_features == 1 and feature_transformation_type == 1:
                    if second_feature_idx not in x_idxs:
                        x_idxs.append(second_feature_idx)
                    param_expression = (
                        (
                            2
                            * np.pi
                            * x[first_feature_idx]
                            * (1 - x[second_feature_idx])
                            - np.pi
                        )
                        * (
                            2
                            * np.pi
                            * x[second_feature_idx]
                            * (1 - x[first_feature_idx])
                            - np.pi
                        )
                        / np.pi
                    )

                if multi_features == 0 and feature_transformation_type == 2:
                    param_expression = (
                        2 * np.arcsin(2 * x[first_feature_idx] - 1) - np.pi
                    )
                if multi_features == 1 and feature_transformation_type == 2:
                    if second_feature_idx not in x_idxs:
                        x_idxs.append(second_feature_idx)
                    param_expression = 2 * np.arcsin(
                        (2 * x[first_feature_idx] - 1) *
                        (2 * x[second_feature_idx] - 1)
                    )

                if gate_list[gate_type_idx] in gate_dict["single_parametric"]:
                    gate(param_expression, k)
                # elif gate_list[gate_type_idx] in gate_dict["two_parametric"]:
                #     gate(param_expression, k, second_qubit_idx)
    return fmap, x_idxs


def genetic_instance(
    gene_space: list[int],
    data_cv: np.ndarray,
    data_labels: np.ndarray,
    backend: StatevectorSimulator,
    gate_dict: dict,
    nb_features: int,
    gates_per_qubits: int,
    nb_qubits: int,
    projected: bool,
    suffix: str,
    coupling_map: list[list[int]],
    basis_gates: list[str],
    fit_fun: Callable[[float, float, int], float],
    noise_std,
    **kwargs: dict,
) -> pygad.GA:
    """Wrapper that returns a genetic instance and initialise time.

    Parameters
    ----------
    gene_space: list[int]
        Span values for all the genes in a chromosome.
    data_cv: np.ndarray
        Training and validation dataset.
    data_labels: np.ndarray
        Training and validation labels.
    backend: StatevectorSimulator
        Backend type (Statevector suggested for speed).
    gate_dict: dict
        Dictionary containing the allowed gates.
    nb_features: int
        Number of features in the dataset.
    gates_per_qubits: int
        Number of gates generated per qubit.
    nb_qubits: int
        Number of qubits for the genetic run.
    projected: bool
        Whether to run standard or projected kernel.
        WARNING: the projected kernel as implemented in this module is extremely slow.
    suffix: str
        Directory and file suffix for saving.
    coupling_map: list[list[str]]
        Backend coupling map.
    basis_gates: list[str]
        List of native gates for the backend.
    fit_fun: Callable[[float, float, int], float]
        Function of the QSVM metrics to return.
    **kwargs: dict
        Other options for the genetic algorithm.

    Returns:
    ----------
    ga_instance: pygad.GA
        Initialised Genetic algorithm instance.
    """
    start_time = time.time()
    ga_instance = pygad.GA(
        fitness_func=fitness_func_wrapper(
            data_cv=data_cv,
            data_labels = data_labels,
            backend = backend,
            gate_dict = gate_dict,
            nb_features = nb_features,
            gates_per_qubits = gates_per_qubits,
            nb_qubits = nb_qubits,
            projected = projected,
            coupling_map = coupling_map,
            basis_gates = basis_gates,
            suffix = suffix,
            fit_fun = fit_fun,
            noise_std = noise_std

        ),
        gene_space=gene_space,
        # parallel_processing = ['thread', 10], # Can be set to have parallelization speedup.
        gene_type=int,
        suppress_warnings=True,
        save_solutions=True,
        on_generation=callback_func_wrapper(start_time),
        **kwargs,
    )
    return ga_instance


def fitness_func_wrapper(
    data_cv: np.ndarray,
    data_labels: np.ndarray,
    backend: StatevectorSimulator,
    gate_dict: dict,
    nb_features: int,
    gates_per_qubits: int,
    nb_qubits: int,
    projected: bool,
    coupling_map: list[list[str]],
    basis_gates: list[str],
    suffix: str,
    fit_fun: Callable[[float, float, int], float],
    noise_std: None
) -> Callable[[pygad.GA, np.ndarray, int], np.float64]:
    """Wrapper that returns a fitness function in a form that pygad.GA instance accepts.

    Parameters
    ----------
    data_cv: np.ndarray
        Training and validation dataset.
    data_labels: np.ndarray
        Training and validation labels.
    backend: StatevectorSimulator
        Backend type (Statevector suggested for speed).
    gate_dict: dict
        Dictionary containing the allowed gates.
    nb_features: int
        Number of features in the dataset.
    gates_per_qubits: int
        Number of gates generated per qubit.
    nb_qubits: int
        Number of qubits for the genetic run.
    projected: bool
        Whether to run standard or projected kernel.
        WARNING: the projected kernel as implemented in this module is extremely slow.
    coupling_map: list[list[str]]
        Backend coupling map.
    basis_gates: list[str]
        List of native gates for the backend.
    suffix: str
        Directory and file suffix for saving.
    fit_fun: Callable[[float, float, int], float]
        Function of the QSVM metrics to return.

    Returns:
    ----------
    fitness_func: Callable[[pygad.GA, np.ndarray, int], np.float64]
        Fitness function that outputs a scalar given the gene sequence that describes a chromosome.
    """

    def fitness_func(ga_instance: pygad.GA, solution: np.ndarray, solution_idx: int) -> Union[np.float64, list[np.float64]]:
        """Computing the fitness function value for a chromosome and saving useful metrics.

        Parameters
        ----------
        ga_instance: pygad.GA
            Initialised Genetic algorithm instance.
        solution: np.ndarray
            Gene sequence describing a chromosome.
        solution_idx: np.ndarray
            Index of the chromosome within its generation.

        Returns:
        ----------
        fitness_value: Union[np.float64, list[np.float64]]
            Fitness value for a chromosome.
        """
        fmap, x_idxs = to_quantum(
            solution, gate_dict, nb_features, gates_per_qubits, nb_qubits
        )

        if 'accuracy' in inspect.signature(fit_fun).parameters:
            mode = "with_kernel"
        else:
            mode = "without_kernel"

        if mode == "with_kernel":
            # Computing a quantum kernel is required. fit_fun evaluation will be slow and return classification accuracy and kernel metrics.
            if projected:
                qker_matrix = projected_quantum_kernel(fmap, data_cv[:, x_idxs], 1)
            else:
                qker = FidelityStatevectorKernel(feature_map=fmap)
                qker_matrix = qker.evaluate(x_vec=data_cv[:, x_idxs])
                if noise_std is not None:
                    mean = 0
                    std_dev = noise_std  # Adjust the standard deviation as needed
                    gaussian_noise = np.random.normal(mean, std_dev, size=qker_matrix.shape)
                    gaussian_noise = np.triu(gaussian_noise, k=1)
                    gaussian_noise = gaussian_noise + gaussian_noise.T
                    noisy_matrix = qker_matrix + gaussian_noise
                    qker_matrix = np.clip(noisy_matrix, 0, 1)


            clf = SVC(kernel="precomputed")

            # param_grid = {"C": [0.01, 0.1, 1, 10, 100, 1000, 10000]}
            # grid_search = GridSearchCV(clf, param_grid, cv=5, scoring="accuracy", verbose=0)
            # grid_search.fit(qker_matrix, data_labels)
            # best_clf = grid_search.best_estimator_
            accuracy_cv_cost = cross_val_score(
                clf,
                qker_matrix,
                data_labels,
                cv=4,
                scoring="accuracy",
            ).mean()
            qker_matrix_0 = qker_matrix[data_labels == 0]
            qker_matrix_0 = np.triu(qker_matrix_0[:, data_labels == 0], 1)
            qker_array_0 = qker_matrix_0[np.triu_indices(qker_matrix_0.shape[0], 1)]
            qker_matrix_1 = qker_matrix[data_labels == 1]
            qker_matrix_1 = np.triu(qker_matrix_1[:, data_labels == 1], 1)
            qker_array_1 = qker_matrix_1[np.triu_indices(qker_matrix_1.shape[0], 1)]

            qker_matrix_01 = qker_matrix[data_labels == 0]
            qker_matrix_01 = qker_matrix_01[:, data_labels == 1]
            fmap_transpiled_depth = transpile(
                fmap, coupling_map=coupling_map, basis_gates=basis_gates
            ).depth()

            sparsity_cost = (np.mean(qker_array_0) + np.mean(qker_array_1)) / 2 - np.mean(
                qker_matrix_01
            )
            offdiagonal_mean = np.mean(np.triu(qker_matrix, 1))
            offdiagonal_std = np.std(np.triu(qker_matrix, 1))
            fitness_value = fit_fun(accuracy_cv_cost, offdiagonal_std, fmap_transpiled_depth)
            print("depth", fmap_transpiled_depth)
            print("sparsity", sparsity_cost)
            print("accuracy", accuracy_cv_cost)
            print("fitness_value", fitness_value)

            save_path = "../../Output_genetic/" + suffix
            Path("../../Output_genetic").mkdir(exist_ok = True)
            Path(save_path).mkdir(exist_ok=True)
            with open(
                save_path + "/genes" + suffix + ".csv", "a", encoding="UTF-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(solution)
            # with open(
            #     save_path + "/kernels_flattened" + suffix + ".csv", "a", encoding="UTF-8"
            # ) as file:
            #     writer = csv.writer(file)
            #     writer.writerow(qker_matrix.reshape(-1))
            with open(
                save_path + "/depth" + suffix + ".txt", "a", encoding="UTF-8"
            ) as file:
                file.write(str(fmap_transpiled_depth) + "\n")
            with open(
                save_path + "/sparsity" + suffix + ".txt", "a", encoding="UTF-8"
            ) as file:
                file.write(str(sparsity_cost) + "\n")
            with open(
                save_path + "/accuracy" + suffix + ".txt", "a", encoding="UTF-8"
            ) as file:
                file.write(str(accuracy_cv_cost) + "\n")
            with open(
                save_path + "/fitness_values_iter_" + suffix + ".csv", "a", encoding="UTF-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(fitness_value) if hasattr(fitness_value, "len") > 1 else writer.writerow([fitness_value])
            with open(
                save_path + "/offdiagonal_mean_" + suffix + ".txt", "a", encoding="UTF-8"
            ) as file:
                file.write(str(offdiagonal_mean) + "\n")
            with open(
                save_path + "/offdiagonal_std_" + suffix + ".txt", "a", encoding="UTF-8"
            ) as file:
                file.write(str(offdiagonal_std) + "\n")
            with open(
                save_path + "/generation_id" + suffix + ".txt", "a", encoding="UTF-8"
            ) as file:
                file.write(str(ga_instance.generations_completed) + "\n")

        else:
            # fit_fun evaluation does not require an explicit kernel evaluation. fit_fun will be fast.
            fitness_value = fit_fun(transpile(fmap, coupling_map = coupling_map, basis_gates = basis_gates, optimization_level=0))
            save_path = "../../Output_genetic/" + suffix
            Path("../../Output_genetic").mkdir(exist_ok = True)
            Path(save_path).mkdir(exist_ok=True)
            with open(
                save_path + "/genes" + suffix + ".csv", "a", encoding="UTF-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(solution)
            with open(
                save_path + "/fitness_values_iter_" + suffix + ".csv", "a", encoding="UTF-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(fitness_value) if hasattr(fitness_value, "len") > 1 else writer.writerow([fitness_value])
        return fitness_value

    return fitness_func


def callback_func_wrapper(start_time) -> Callable[[pygad.GA], None]:
    """Wrapper that returns a callback function in a form that pygad.GA instance accepts.

    Parameters
    ----------
    start_time: np.float64
        Starting time of the genetic run.

    Returns:
    ----------
    callback_func: Callable[[pygad.GA], None]
        Callback function called at the end of every generation (except for generation 0).
    """

    def callback_func(ga_instance: pygad.GA) -> None:
        """Callback function that prints useful information at the end of every generation (except for generation 0).

        Parameters
        ----------
        ga_instance: pygad.GA
            Genetic instance class.

        Returns:
        ----------
        None
        """
        fitness_values = ga_instance.last_generation_fitness.max()
        print("Generation:", ga_instance.generations_completed)
        print(f"Best fitness: {fitness_values}")
        end_time = time.time()
        print("Elapsed time: " + str(end_time - start_time) + "s")

    return callback_func


def projected_quantum_kernel(
    fmap: QuantumCircuit, dataset: np.ndarray, gamma: float
) -> np.ndarray:
    """Returns a one-particle reduced density matrix (1-RDM) projected quantum kernel matrix
    as described here: https://www.nature.com/articles/s41467-021-22539-9.
    and further documented here: https://www.researchsquare.com/article/rs-2296310/v1.

    This function is roughly 10 times slower than the standard kernel evaluation in qiskit.
    There might be margin for improvement.

    Parameters
    ----------
    fmap: QuantumCircuit
        Quantum featuremap.
    dataset: np.ndarray
        Training and validation dataset.
    gamma: positive hyperparameter.

    Returns:
    ----------
    kernel_matrix: np.ndarray
        Projected quantum kernel matrix.
    """
    if not fmap.parameters:
        kernel_matrix = np.ones((dataset.shape[0], dataset.shape[0]))
        return kernel_matrix
    kernel_matrix = np.zeros((dataset.shape[0], dataset.shape[0]))
    for i in range(dataset.shape[0]):
        for j in range(i):
            statevector_i_dm = DensityMatrix(fmap.assign_parameters(dataset[i]))
            statevector_j_dm = DensityMatrix(fmap.assign_parameters(dataset[j]))
            exp_term = 0
            for q in range(fmap.num_qubits):
                summed_qubits = [k for k in range(fmap.num_qubits) if k != q]
                exp_term = exp_term + np.linalg.norm(
                    partial_trace(statevector_i_dm, summed_qubits)
                    - partial_trace(statevector_j_dm, summed_qubits)
                )
            kernel_matrix[i, j] = np.exp(-gamma * exp_term)
    kernel_matrix = kernel_matrix + kernel_matrix.T + np.identity(dataset.shape[0])
    return kernel_matrix
