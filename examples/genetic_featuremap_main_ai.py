import numpy as np
from pathlib import Path
from quake.models.qsvm.qsvm_tester import get_features
from qiskit import QuantumCircuit, transpile, Aer, execute
import pygad


def create_random_feature_map(num_qubits, max_gates_per_qubit, gate_dict):
    feature_map = QuantumCircuit(num_qubits)

    for qubit in range(num_qubits):
        num_gates = np.random.randint(1, max_gates_per_qubit + 1)
        for _ in range(num_gates):
            gate_type = np.random.choice(list(gate_dict.keys()))
            gate = np.random.choice(list(gate_dict[gate_type]))
            if gate_type == "single_non_parametric":
                feature_map.append(gate, [qubit])
            elif gate_type == "single_parametric":
                angle = np.random.rand() * 2 * np.pi
                feature_map.append(gate(angle), [qubit])
            elif gate_type == "two_non_parametric":
                target = np.random.randint(0, num_qubits)
                if target != qubit:
                    feature_map.append(gate, [qubit, target])
            elif gate_type == "two_parametric":
                angle = np.random.rand() * 2 * np.pi
                target = np.random.randint(0, num_qubits)
                if target != qubit:
                    feature_map.append(gate(angle), [qubit, target])

    return feature_map


def evaluate_feature_map(feature_map, dataset, labels, num_shots=1000):
    backend = Aer.get_backend('qasm_simulator')
    shots = num_shots if num_shots > 1 else 1
    circuit = transpile(feature_map, backend=backend)
    job = execute(circuit, backend=backend, shots=shots)
    result = job.result().get_counts(circuit)

    accuracy = 0
    for data, label in zip(dataset, labels):
        predicted_label = 1 if result.get(data, 0) >= shots / 2 else 0
        if predicted_label == label:
            accuracy += 1
    accuracy /= len(dataset)

    return accuracy


def fitness_function(solution, dataset, labels):
    feature_map = create_feature_map(solution)
    accuracy = evaluate_feature_map(feature_map, dataset, labels)
    return accuracy


def create_feature_map(gene_space_solution):
    num_qubits, max_gates_per_qubit, gate_dict = gene_space_solution
    feature_map = create_random_feature_map(num_qubits, max_gates_per_qubit, gate_dict)
    return feature_map


def genetic_algorithm_search(dataset, labels, gene_space, num_generations=50, num_parents_mating=4, fitness_func=fitness_function):
    num_genes = len(gene_space)
    sol_per_pop = 10
    num_generations = num_generations
    num_parents_mating = num_parents_mating

    initial_population = []
    for _ in range(sol_per_pop):
        solution = [np.random.choice(gene_space[gene]) for gene in gene_space]
        initial_population.append(solution)

    num_generations = num_generations
    num_parents_mating = num_parents_mating

    best_outputs = []
    best_solutions = []

    def callback_generation(ga_instance):
        best_outputs.append(ga_instance.best_solution()[1])
        best_solutions.append(ga_instance.best_solution()[0])

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           initial_population=initial_population,
                           callback_generation=callback_generation)

    ga_instance.run()

    best_solution = ga_instance.best_solution()[0]
    best_feature_map = create_feature_map(*best_solution)

    return best_feature_map, best_solution




from quake.utils.utils import load_runcard, save_runcard
from quake.models.autoencoder.autoencoder_dataloading import read_data
from quake.models.autoencoder.train import load_and_compile_network
import numpy as np
from pathlib import Path
from quake.models.qsvm.qsvm_tester import get_features


if __name__ == "__main__":
    NUM_QUBITS = 5
    MAX_GATES_PER_QUBIT = 6
    GATE_DICT = {
        "single_non_parametric": {"I", "H", "X", "S"},
        "single_parametric": {"RX", "RY", "RZ"},
        "two_non_parametric": {"CX"},
        "two_parametric": {"CRX", "CRY", "CRZ", "CP"},
    }

    # Dataset loading
    data_folder = Path("../../output_2/data")
    train_folder = Path("../../output_2/models/autoencoder")
    setup = load_runcard("../../output_2/cards/runcard.yaml")
    setup["run_tf_eagerly"] = False
    setup["seed"] = 42

    dataset, labels = get_features(data_folder.parent, "autoencoder", setup)
    dataset = dataset[0][:300]
    labels = labels[0][:300]
    gene_space = {
        "num_qubits": [NUM_QUBITS],
        "max_gates_per_qubit": list(range(1, MAX_GATES_PER_QUBIT + 1)),
        "gate_dict": [GATE_DICT],
    }

    best_feature_map, best_solution = genetic_algorithm_search(dataset, labels, gene_space)

    print("Best Quantum Feature Map:")
    print(best_feature_map)
    print("Best Solution:")
    print(best_solution)
