""" This module contains the functions for generating a genetic-optimized quantum featuremap. """

from qiskit.circuit import QuantumCircuit
import numpy as np
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.kernels import QuantumKernel
from pathlib import Path
from qiskit import Aer
from sklearn.model_selection import train_test_split
from typing import Tuple
from sklearn.svm import SVC
import random
import pickle
import pygad

x = ParameterVector("x", 2)
backend = Aer.get_backend("statevector_simulator")


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


def chunks(bits: np.ndarray) -> list([np.ndarray]):
    """Returns a list of input array slices of length 6

    Parameters
    ----------
    bits: np.ndarray
        Input array to slice.

    Returns
    -------
    bits_chunks: list([np.ndarray])
        Sliced array.
    """
    bits_chunks = []
    n = 6
    for i in range(0, len(bits), n):
        bits_chunks.append(bits[i : i + n])
    return bits_chunks


def match_gate(
    fmap: QuantumCircuit, binary_block: np.ndarray, q: int, x: ParameterVector
) -> QuantumCircuit:
    """Appends a new gate to a quantum featuremap by decoding the bitstring.

    Parameters
    ----------
    fmap: QuantumCircuit
        Input quantum featuremap.
    binary_block: np.ndarray
        Bitstring containing information on the encoding and the quantum gate.
    q: int
        Quantum registry index (on which qubit-line append the gate).
    x: ParameterVector
        Free parameters of the quantum circuits.

    Returns
    -------
    fmap: QuantumCircuit
        Output quantum featuremap.
    """
    binary_gate = binary_block[:3]
    binary_feature = binary_block[3:]

    if np.array_equal(binary_feature, [0, 0, 0]):
        func = 2 * x[0]
    if np.array_equal(binary_feature, [0, 0, 1]):
        func = 2 * x[1]
    if np.array_equal(binary_feature, [0, 1, 0]):
        func = np.arcsin(2 / np.pi * x[0])
    if np.array_equal(binary_feature, [0, 1, 1]):
        func = np.arccos(4 / np.pi / np.pi * x[0] * x[0])
    if np.array_equal(binary_feature, [1, 0, 0]):
        func = (np.pi - 2 * x[0]) * (np.pi - 2 * x[1])
    if np.array_equal(binary_feature, [1, 0, 1]):
        func = 4 * x[0] * x[1]
    if np.array_equal(binary_feature, [1, 1, 0]):
        func = np.arcsin(2 / np.pi * x[1])
    if np.array_equal(binary_feature, [1, 1, 1]):
        func = np.arccos(4 / np.pi / np.pi * x[1] * x[1])

    if np.array_equal(binary_gate, [0, 0, 0]):
        return fmap
    if np.array_equal(binary_gate, [0, 0, 1]):
        fmap.h(q)
        return fmap
    if np.array_equal(binary_gate, [0, 1, 0]):
        fmap.cx(0, 1)
        return fmap
    if np.array_equal(binary_gate, [0, 1, 1]):
        fmap.rz(func, q)
        return fmap
    if np.array_equal(binary_gate, [1, 0, 0]):
        fmap.rx(func, q)
        return fmap
    if np.array_equal(binary_gate, [1, 0, 1]):
        fmap.ry(func, q)
        return fmap
    if np.array_equal(binary_gate, [1, 1, 0]):
        fmap.p(func, q)
        return fmap
    if np.array_equal(binary_gate, [1, 1, 1]):
        return fmap


def initial_population(n_fmap: int) -> list(list([int])):
    """Returns n random binary encoded featuremaps.

    Parameters
    ----------
    n_fmap: int
        Number of featuremaps to initialize.

    Returns
    -------
    initial_pop: list(list([int]))
        List of binary featuremaps.
    """
    initial_pop = []
    for j in range(n_fmap):
        initial_pop.append([random.randrange(0, 2) for i in range(36)])

    return initial_pop


def to_quantum(bitlist: np.ndarray) -> QuantumCircuit:
    """Converts a bitstring to a corresponding quantum featuremap.

    Parameters
    ----------
    bitlist: np.ndarray
        Binary encoded featuremap

    Returns
    -------
    fmap: QuantumCircuit
        Quantum featuremap.
    """
    bits_q0 = chunks(bitlist[:18])
    bits_q1 = chunks(bitlist[18:])
    fmap = QuantumCircuit(2)
    for gate_q0, gate_q1 in zip(bits_q0, bits_q1):
        fmap = match_gate(fmap, gate_q0, 0, x)
        fmap = match_gate(fmap, gate_q1, 1, x)
    return fmap


def quick_comparison(
    fittest_fmap: QuantumCircuit,
    classic_kernels: list([str]),
    data_train: np.ndarray,
    lab_train: np.ndarray,
    data_compare: np.ndarray,
    lab_compare: np.ndarray,
):
    """Trains SVM and QSVM with the fittest kernel, printing the scores of each on a validation dataset.

    Parameters
    ----------
    fittest_fmap: QuantumCircuit
        The output featuremap of the genetic algorithm.
    classic_kernels: list([str])
        List of classical kernels. Available options: {'linear', 'poly', 'rbf', 'sigmoid'}.
    data_train: np.ndarray
        Training dataset.
    lab_train: np.ndarray
        Training truth labels.
    data_compare: np.ndarray
        Validation dataset.
    lab_compare: np.ndarray
        Validation truth labels.
    """
    qker = QuantumKernel(feature_map=fittest_fmap, quantum_instance=backend)
    qker_matrix_train = qker.evaluate(x_vec=data_train)
    qker_matrix_compare = qker.evaluate(y_vec=data_train, x_vec=data_compare)
    clf = SVC(kernel="precomputed").fit(qker_matrix_train, lab_train)
    qt_accuracy = clf.score(qker_matrix_compare, lab_compare)
    print(f"Quantum kernel accuracy is: {qt_accuracy}")

    for ker in classic_kernels:
        clf = SVC(kernel=ker).fit(data_train, lab_train)
        accuracy = clf.score(data_compare, lab_compare)
        print(f"{ker} accuracy is: {accuracy}")


def save_results(fittest_fmap: QuantumCircuit, ga_instance: pygad.GA, foldername: Path):
    """Saves the progress of the genetic optimization through epochs, the fittest featuremap, a plot of the fitness function through the epochs.

    Parameters
    ----------
    fittest_fmap: QuantumCircuit
        The output featuremap of the genetic algorithm.
    ga_instance: pygad.GA
        Genetic algorithm instance.
    foldername: Path
        Directory where files will be saved.
    """
    foldername.mkdir(exist_ok=True)
    with open(foldername / Path("gen_fmap.pkl"), "wb") as f:
        pickle.dump(fittest_fmap, f)
    with open(foldername / Path("ga_instance.pkl"), "wb") as f:
        ga_instance.fitness_func = None
        pickle.dump(ga_instance, f)

    fitness_plot = ga_instance.plot_fitness()
    fitness_plot.savefig(foldername / Path("fitness_plot.png"))


def genetic_instance(
    opts: dict,
    data_train: np.ndarray,
    lab_train: np.ndarray,
    data_val: np.ndarray,
    lab_val: np.ndarray,
) -> pygad.GA:
    """Return a genetic instance ready to run.

    Parameters
    ----------
    opts: dict
        Customizable input options.
    data_train: np.ndarray
        Training dataset.
    lab_train: np.ndarray
        Training truth labels.
    data_compare: np.ndarray
        Validation dataset.
    lab_compare: np.ndarray
        Validation truth labels.

    Returns
    -------
    ga_instance: pygad.GA
        The genetic instance.
    """

    def fitness_func(solution: np.ndarray, solution_idx: int) -> np.float64:
        """Converts the binary featuremap into a quantum circuit and return its classification score.

        Parameters
        ----------
        solution: np.ndarray
            The binary featuremap to evaluate.
        solution_idx: int
            Index of the sample in the population.

        Returns
        -------
        accuracy: np.float64
            The fitness value, corresponding to the classification score on a validation dataset.
        """
        fmap = to_quantum(solution)
        print(fmap)
        if fmap.num_parameters == 2:
            fmap.assign_parameters({x: [0, 0]})
            qker = QuantumKernel(feature_map=fmap, quantum_instance=backend)
            qker_matrix_train = qker.evaluate(x_vec=data_train)
            qker_matrix_val = qker.evaluate(y_vec=data_train, x_vec=data_val)
            clf = SVC(kernel="precomputed", C=100).fit(qker_matrix_train, lab_train)
            accuracy = clf.score(qker_matrix_val, lab_val)
        else:
            accuracy = 0.0
        print(f"Fitness value: {accuracy}")
        return accuracy

    ga_instance = pygad.GA(
        fitness_func=fitness_func,
        init_range_low=0,
        init_range_high=2,
        random_mutation_min_val=0,
        random_mutation_max_val=2,
        gene_type=int,
        **opts,
    )
    return ga_instance
