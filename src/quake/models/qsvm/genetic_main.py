""" This module contains the functions for generating a genetic-optimized quantum featuremap. 
This module uses integer encoding instead of binary encoding and allows for more search options than 'genetic.py' and 'genetic_extended.py'"""

from qiskit.circuit import QuantumCircuit
import numpy as np
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.kernels import QuantumKernel
from pathlib import Path
from qiskit import Aer
from sklearn.model_selection import train_test_split
from typing import Tuple
from sklearn.svm import SVC
import pickle
import pygad

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

def initial_population(nb_init_individuals, nb_qubits, gates_per_qubits, gate_dict, nb_features):
    
    nb_possible_gates = len(gate_dict["single_non_parametric"]) + len(gate_dict["single_parametric"]) + len(gate_dict["two_non_parametric"]) + len(gate_dict["two_parametric"]) 
    
    size_per_gene = nb_qubits*gates_per_qubits*nb_init_individuals
    gate_idxs = gen_int(0, nb_possible_gates, size = size_per_gene)
    feature_transformation = gen_int(0, 2, size = size_per_gene)
    multi_features = gen_int(0, 2, size = size_per_gene)
    first_feature_idx = gen_int(0, nb_features, size = size_per_gene)
    second_feature_idx = gen_int(0, nb_features, size = size_per_gene, exclude_array = first_feature_idx)
    gene_list = np.array([gate_idxs, feature_transformation, multi_features, first_feature_idx, second_feature_idx])
    gene_list = np.reshape(gene_list.T, [nb_init_individuals, gates_per_qubits, nb_qubits, 5])                
    return gene_list

def gen_int(min_val, max_val, size = None, exclude_array = None):
    if exclude_array is not None:
        valid_values_per_index = [np.setdiff1d(np.arange(min_val, max_val), [exclude_array[i]]) for i in range(size)]
        random_indices = [np.random.choice(valid_values_per_index[i]) for i in range(size)]
    else:
        random_indices = np.random.randint(min_val, max_val, size)
    return random_indices

def interpret_gate(qc, gate_string):
    return getattr(qc, gate_string.lower())


def to_quantum(genes, gate_dict):

    gate_list = []
    for gate_set in gate_dict.values():
        gate_list = gate_list + list(gate_set)

    fmap_list = []
    gates_per_qubits = genes[0].shape[0]
    nb_qubits = genes[0].shape[1]

    for i, chromosome in enumerate(genes):
        fmap = QuantumCircuit(nb_qubits)
        for j in range(gates_per_qubits):
            for k in range(nb_qubits):
                gate_type_idx = genes[i, j, k, 0]
                feature_transformation = genes[i, j, k, 1]
                multi_features = genes[i, j, k, 2]
                first_feature_idx = genes[i, j, k, 3]
                second_feature_idx = genes[i, j, k, 4]
                # If necessary, call also this:
                # second_qubit_idx = gen_int(0, nb_qubits, size = 1, exclude_array = [k])[0]
                gate = interpret_gate(fmap, gate_list[gate_type_idx])
                # calling gate(x, q1, q2) adds the gate to fmap

    return fmap_list