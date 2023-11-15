""" This module contains the functions for generating a genetic-optimized quantum featuremap. 
This module uses integer encoding instead of binary encoding and allows for more search options than 'genetic.py' and 'genetic_extended.py'"""

from qiskit.circuit import QuantumCircuit
import numpy as np
import math
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.kernels import QuantumKernel
from pathlib import Path
from qiskit import Aer
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import partial_trace, DensityMatrix
from qiskit.providers.fake_provider import FakeLagosV2
from qiskit.compiler import transpile

from sklearn.model_selection import train_test_split
from typing import Tuple
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import pickle
import pygad
import os
import time
import h5py

import csv


def get_subsample(
    dataset: np.ndarray, labels: np.ndarray, size: int, seed: int = 42
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

    Returns
    -------
    subs_dataset: np.ndarray
        Subsampling of the feature distribution.
    subs_labels: np.ndarray
        Subsample truth labels.
    """
    subs_dataset, subs_labels = train_test_split(
        dataset, labels, train_size=size, random_state=seed
    )[::2]
    return subs_dataset, subs_labels


def initial_population(
    nb_init_individuals: int, nb_qubits: int, gates_per_qubits: int, gate_dict: dict, nb_features: int
):
    nb_possible_gates = (
        len(gate_dict["single_non_parametric"])
        + len(gate_dict["single_parametric"])
        + len(gate_dict["two_non_parametric"])
        + len(gate_dict["two_parametric"])
    )

    size_per_gene = nb_qubits * gates_per_qubits * nb_init_individuals
    gate_idxs = gen_int(0, nb_possible_gates, size=size_per_gene)
    feature_transformation = gen_int(0, 2, size=size_per_gene)
    multi_features = gen_int(0, 2, size=size_per_gene)
    first_feature_idx = gen_int(0, nb_features, size=size_per_gene)
    second_feature_idx = gen_int(
        0, nb_features, size=size_per_gene, exclude_array=first_feature_idx
    )    
    second_qubit_idx = gen_int(0, nb_qubits, size=size_per_gene, exclude_array=np.tile(np.arange(0, nb_qubits), gates_per_qubits * nb_init_individuals))
    gene_list = np.array(
        [
            gate_idxs,
            feature_transformation,
            multi_features,
            first_feature_idx,
            second_feature_idx,
            second_qubit_idx
        ]
    )
    gene_list = np.reshape(
        gene_list.T, [nb_init_individuals, gates_per_qubits, nb_qubits, 6]
    )
    gene_list_flat = flatten_gene_list(gene_list)
    return gene_list_flat


def flatten_gene_list(gene_list):
    population_size = gene_list.shape[0]
    gene_list_flat = gene_list.reshape(population_size, -1)
    return gene_list_flat


def unflatten_gene_list(
    gene_list_flat, nb_init_individuals, gates_per_qubits, nb_qubits
):
    gene_list = np.reshape(
        gene_list_flat, [nb_init_individuals, gates_per_qubits, nb_qubits, 6]
    )
    return gene_list


def get_gene_space(gate_dict, nb_features, nb_qubits, gates_per_qubits):
    nb_possible_gates = (
        len(gate_dict["single_non_parametric"])
        + len(gate_dict["single_parametric"])
        + len(gate_dict["two_non_parametric"])
        + len(gate_dict["two_parametric"])
    )
    size_per_gene = nb_qubits * gates_per_qubits
    gene_space = []
    for i in range(size_per_gene):
        gene_space = gene_space + [
            range(nb_possible_gates),
            range(2),
            range(2),
            range(nb_features),
            range(nb_features),
            range(nb_qubits)
        ]
    return gene_space


def gen_int(min_val, max_val, size=None, exclude_array=None):
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


def interpret_gate(qc, gate_string):
    return getattr(qc, gate_string.lower())


def to_quantum(genes, gate_dict, nb_features, gates_per_qubits, nb_qubits):
    gate_list = []
    for gate_set in gate_dict.values():
        gate_list = gate_list + list(gate_set)

    genes_unflatted = np.reshape(genes, [gates_per_qubits, nb_qubits, 6])
    # gates_per_qubits = genes[0].shape[0]
    # nb_qubits = genes[0].shape[1]
    x = ParameterVector("x", length=nb_features)
    fmap = QuantumCircuit(nb_qubits)
    x_idxs = []
    for j in range(gates_per_qubits):
        for k in range(nb_qubits):
            gate_type_idx = genes_unflatted[j, k, 0]
            feature_transformation_type = genes_unflatted[j, k, 1]
            multi_features = genes_unflatted[j, k, 2]
            first_feature_idx = genes_unflatted[j, k, 3]
            second_feature_idx = genes_unflatted[j, k, 4]
            second_qubit_idx = genes_unflatted[j, k, 5]

            # If necessary, call also this:
            # second_qubit_idx = gen_int(0, nb_qubits, size = 1, exclude_array = [k])[0]
            gate = interpret_gate(fmap, gate_list[gate_type_idx])

            if gate_list[gate_type_idx] in gate_dict["single_non_parametric"]:
                gate(k)
            elif gate_list[gate_type_idx] in gate_dict["two_non_parametric"]:
                # second_qubit_idx = gen_int(0, nb_qubits, size=1, exclude_array=[k])[0]
                gate(k, second_qubit_idx)
            else:
                if first_feature_idx not in x_idxs:
                    x_idxs.append(first_feature_idx)

                if multi_features == 0 and feature_transformation_type == 0:
                    param_expression = x[first_feature_idx]
                if multi_features == 1 and feature_transformation_type == 0:
                    if second_feature_idx not in x_idxs:
                        x_idxs.append(second_feature_idx)
                    param_expression = (2*np.pi - x[first_feature_idx]) * (
                        2*np.pi - x[second_feature_idx]
                    ) / (2*np.pi)
                if multi_features == 0 and feature_transformation_type == 1:
                    param_expression = x[first_feature_idx] * x[first_feature_idx] / (2*np.pi)
                if multi_features == 1 and feature_transformation_type == 1:
                    if second_feature_idx not in x_idxs:
                        x_idxs.append(second_feature_idx)
                    param_expression = (2*np.pi - x[first_feature_idx]*x[first_feature_idx]/ (2*np.pi)) * (
                        2*np.pi - x[second_feature_idx]*x[second_feature_idx]/ (2*np.pi)
                    ) / (2*np.pi)

                if gate_list[gate_type_idx] in gate_dict["single_parametric"]:
                    gate(param_expression, k)
                elif gate_list[gate_type_idx] in gate_dict["two_parametric"]:
                    # second_qubit_idx = gen_int(0, nb_qubits, size=1, exclude_array=[k])[
                    #     0
                    # ]
                    gate(param_expression, k, second_qubit_idx)
    return fmap, x_idxs


# def to_quantum_batch(genes, gate_dict, nb_features, gates_per_qubits, nb_qubits):
#     gate_list = []
#     for gate_set in gate_dict.values():
#         gate_list = gate_list + list(gate_set)

#     fmap_list = []

#     genes_unflatted = unflatten_gene_list(
#         genes, len(genes), gates_per_qubits, nb_qubits
#     )
#     # gates_per_qubits = genes[0].shape[0]
#     # nb_qubits = genes[0].shape[1]

#     x = ParameterVector("x", length=nb_features)
#     for i, chromosome in enumerate(genes_unflatted):
#         fmap = QuantumCircuit(nb_qubits)
#         for j in range(gates_per_qubits):
#             for k in range(nb_qubits):
#                 gate_type_idx = genes_unflatted[i, j, k, 0]
#                 feature_transformation_type = genes_unflatted[i, j, k, 1]
#                 multi_features = genes_unflatted[i, j, k, 2]
#                 first_feature_idx = genes_unflatted[i, j, k, 3]
#                 second_feature_idx = genes_unflatted[i, j, k, 4]
#                 # If necessary, call also this:
#                 # second_qubit_idx = gen_int(0, nb_qubits, size = 1, exclude_array = [k])[0]
#                 gate = interpret_gate(fmap, gate_list[gate_type_idx])

#                 if gate_list[gate_type_idx] in gate_dict["single_non_parametric"]:
#                     gate(k)
#                 elif gate_list[gate_type_idx] in gate_dict["two_non_parametric"]:
#                     second_qubit_idx = gen_int(0, nb_qubits, size=1, exclude_array=[k])[
#                         0
#                     ]
#                     gate(k, second_qubit_idx)
#                 else:
#                     if multi_features == 0 and feature_transformation_type == 0:
#                         param_expression = x[first_feature_idx]
#                     if multi_features == 1 and feature_transformation_type == 0:
#                         param_expression = (np.pi - x[first_feature_idx]) * (
#                             np.pi - x[second_feature_idx]
#                         )
#                     if multi_features == 0 and feature_transformation_type == 1:
#                         param_expression = x[first_feature_idx] * x[first_feature_idx]
#                     if multi_features == 1 and feature_transformation_type == 1:
#                         param_expression = x[first_feature_idx] * x[second_feature_idx]

#                     if gate_list[gate_type_idx] in gate_dict["single_parametric"]:
#                         gate(param_expression, k)
#                     elif gate_list[gate_type_idx] in gate_dict["two_parametric"]:
#                         second_qubit_idx = gen_int(
#                             0, nb_qubits, size=1, exclude_array=[k]
#                         )[0]
#                         gate(param_expression, k, second_qubit_idx)
#         fmap_list.append(fmap)
#     return fmap_list


def genetic_instance(
    opts: dict,
    gene_space,
    data_cv: np.ndarray,
    data_labels: np.ndarray,
    backend,
    gate_dict,
    nb_features,
    gates_per_qubits,
    nb_qubits,
    projected,
    suffix,
    coupling_map,
    basis_gates
) -> pygad.GA:
    start_time = time.time()
    ga_instance = pygad.GA(
        fitness_func=fitness_func_wrapper(
            data_cv,
            data_labels,
            backend,
            gate_dict,
            nb_features,
            gates_per_qubits,
            nb_qubits,
            projected,
            coupling_map,
            basis_gates,
            suffix
        ),
        # init_range_low=0,
        # init_range_high=2,
        gene_space=gene_space,
        # parallel_processing = ['thread', 10],
        # random_mutation_min_val=0,
        # random_mutation_max_val=2,
        gene_type=int,
        suppress_warnings=True,
        save_solutions=True,
        on_generation=callback_func_wrapper(suffix, start_time),
        **opts,
    )
    return ga_instance


def fitness_func_wrapper(
    data_cv, data_labels, backend, gate_dict, nb_features, gates_per_qubits, nb_qubits, projected, coupling_map, basis_gates, suffix
):
    def fitness_func(ga_instance, solution: np.ndarray, solution_idx: int) -> np.float64:
        fmap, x_idxs = to_quantum(solution, gate_dict, nb_features, gates_per_qubits, nb_qubits )
        print(solution)
        import pdb; pdb.set_trace()
        if projected:
            qker_matrix = projected_quantum_kernel(fmap, data_cv[:, x_idxs], 1)
        else:
            qker = QuantumKernel(feature_map=fmap, quantum_instance=backend)
            qker_matrix = qker.evaluate(x_vec=data_cv[:, x_idxs])

        clf = SVC(kernel="precomputed")

        param_grid = {"C": [0.01, 0.1, 1, 10, 100, 1000, 10000]}
        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring="accuracy", verbose = 0)
        grid_search.fit(qker_matrix, data_labels)
        best_clf = grid_search.best_estimator_
        accuracy_cv_cost = cross_val_score(
            best_clf, qker_matrix, data_labels, cv=5, scoring="accuracy",
        ).mean()
        qker_matrix_0 = qker_matrix[data_labels == 0]
        qker_matrix_0 = np.triu(qker_matrix_0[:, data_labels == 0], 1)
        qker_matrix_1 = qker_matrix[data_labels == 1]
        qker_matrix_1 = np.triu(qker_matrix_1[:, data_labels == 1], 1)

        qker_matrix_01 = qker_matrix[data_labels == 0]
        qker_matrix_01 = qker_matrix_01[:,data_labels == 1]
        fmap_transpiled_depth = transpile(fmap, coupling_map = coupling_map, basis_gates = basis_gates).depth()
        # fitness_value = np.mean(qker_matrix_0[qker_matrix_0 > 0]) + np.mean(qker_matrix_1[qker_matrix_1 > 0]) - np.mean(qker_matrix_01)
        sparsity_cost = (np.sum(qker_matrix_0) + np.sum(qker_matrix_1)) / (qker_matrix_0.shape[0]**2/2 - qker_matrix_0.shape[0] + qker_matrix_1.shape[0]**2/2 - qker_matrix_1.shape[0])
        fitness_value = accuracy_cv_cost #+ -1e-5*fmap_transpiled_depth + 1e-3* sparsity_cost #1e-2*depth_cost + 1e-2*sparsity_cost + np.exp(1+accuracy_cv_cost)
        # print("depth", fmap_transpiled_depth)
        # print("sparsity", sparsity_cost)
        # print("accuracy", accuracy_cv_cost)
        with open("depth" + suffix + ".txt", "a") as file:
            file.write(str(fmap_transpiled_depth) + "\n")
        with open("sparsity" + suffix + ".txt", "a") as file:
            file.write(str(sparsity_cost) + "\n")
        with open("accuracy" + suffix + ".txt", "a") as file:
            file.write(str(accuracy_cv_cost) + "\n")
        with open("fitness_values_iter_" + suffix + ".txt", "a") as file:
            file.write(str(fitness_value) + "\n")
        print(fmap)
        return fitness_value
    return fitness_func

def callback_func_wrapper(suffix, start_time):
    def callback_func(ga_instance):
        fitness_file_name = "fitness_values_"
        kernel_file_name = "kernels_"
        best_kernels = "best_kernels_"
        if ga_instance.generations_completed == 1 and os.path.isfile(
            fitness_file_name + suffix + ".txt"
        ):
            raise Exception(
                "File "
                + fitness_file_name
                + suffix
                + ".txt"
                + " already exists. Rename or delete it."
            )
        fitness_values = ga_instance.last_generation_fitness
        kenel_bitstrings = ga_instance.solutions
        print("Generation:", ga_instance.generations_completed)
        print("Best fitness: "+ str(np.max(fitness_values)))
        print("Avg. fitness: "+ str(np.mean(fitness_values)))
        print("Std. fitness: "+ str(np.std(fitness_values)))
        end_time = time.time()
        print("Elapsed time: " + str(end_time - start_time) + "s")
        # print(ga_instance.best_solution())
        with open(fitness_file_name + suffix + ".txt", 'a') as file:
            np.savetxt(file, np.array(fitness_values))
        # with open(best_kernels + suffix + ".txt", 'a') as file:
        #     np.savetxt(file, np.array(ga_instance.best_solution()[0]))    
        np.save(kernel_file_name+suffix, np.array(kenel_bitstrings))

    return callback_func

def projected_quantum_kernel(fmap, dataset, gamma):
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
                exp_term = exp_term + np.linalg.norm(partial_trace(statevector_i_dm, summed_qubits) - partial_trace(statevector_j_dm, summed_qubits))
            kernel_matrix[i, j] = np.exp(-gamma*exp_term)
    kernel_matrix = kernel_matrix + kernel_matrix.T + np.identity(dataset.shape[0])
    return kernel_matrix    
