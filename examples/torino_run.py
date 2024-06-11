"""We execute 21 identical qsvms to study site performance"""
import numpy as np
from pathlib import Path
from qiskit_ibm_runtime import Session

from collections import OrderedDict
from typing import Union
from qiskit.circuit import QuantumCircuit
from qiskit.compiler import transpile
from quake.models.qsvm.genetic_v5 import to_quantum, get_kernel_entry
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import csv
from qiskit_ibm_provider import IBMProvider
from quake.models.qsvm import genetic_v5 as genetic

###################################################################
############# LOAD AN ACCOUNT THAT CAN RUN ON TORINO ##############
###################################################################

IBMProvider.save_account(token='', overwrite=True)
provider = IBMProvider(instance='')
backend = provider.get_backend('ibm_torino')

###################################################################
###################################################################
###################################################################

qsvm_connections = [    
    [0, 1, 2, 3],
    [5, 6, 7, 8],
    [10, 11, 12, 13],
    [19, 20, 21, 34],
    [16, 23, 24, 25],
    [29, 30, 31, 32],
    [54, 42, 43, 44],
    [55, 46, 47, 48],
    [37, 52, 51, 50],
    [57, 58, 59, 60],
    [66, 67, 68, 69],
    [71, 75, 90, 89],
    [84, 85, 86, 87],
    [79, 80, 81, 82],
    [77, 76, 91, 95],
    [97, 98, 99, 100],
    [102, 103, 104, 105],
    [107, 108, 109, 113],
    [129, 114, 115, 116],
    [119, 120, 121, 122],
    [124, 125, 126,127]
    ]
gate_dict = OrderedDict(
    [
        ("single_non_parametric", ["Id", "X","SX"]),
        ("single_parametric", ["RZ", "RZ"]),
        ("two_non_parametric", ["CZ"]),
        ("two_parametric", []),
    ]
)
nb_qubits = 4
gates_per_qubits = 8
data_cv, data_labels = np.load("../processed_dataset/data_cv.npy"), np.load("../processed_dataset/data_labels.npy")
useful_ft = [0, 5, 6, 8, 10, 12, 14, 15]
data_cv = data_cv[:, useful_ft]

nb_features = data_cv.shape[1]
nb_cbits = 4*21
cbits = [item for item in range(0, nb_cbits)]
nb_samples = 100
genes = [3, 0, 1, 2, 5, 2, 1, 0, 1, 6, 3, 2, 1, 0, 2, 4, 1, 1, 0, 4, 5, 1,
       1, 5, 2, 2, 1, 1, 4, 1, 4, 2, 1, 3, 6, 4, 2, 1, 6, 3, 4, 0, 1, 3,
       1, 2, 0, 1, 0, 1, 1, 2, 1, 3, 1, 2, 2, 1, 4, 6, 5, 1, 0, 4, 2, 5,
       2, 1, 2, 1, 2, 0, 1, 0, 4, 2, 1, 1, 2, 7, 3, 0, 0, 0, 3, 3, 1, 1,
       3, 4, 1, 0, 0, 0, 5, 2, 2, 1, 0, 1, 0, 1, 0, 5, 6, 4, 0, 1, 2, 0,
       3, 1, 0, 0, 7, 4, 1, 0, 4, 6, 1, 1, 0, 3, 6, 1, 1, 1, 6, 7, 2, 0,
       1, 4, 3, 5, 2, 1, 7, 1, 1, 2, 0, 2, 1, 1, 0, 0, 3, 5, 3, 2, 1, 1,
       7, 3, 1, 0, 5, 4]

generation_zero = genetic.initial_population(
    1, nb_qubits, gates_per_qubits, gate_dict, nb_features
)
genes = generation_zero[0]


suffix = 'torino_run'
save_path = "../../Output_genetic/" + suffix
Path("../../Output_genetic").mkdir(exist_ok=True)
Path(save_path).mkdir(exist_ok=True)
with open(
    save_path + "/genes" + suffix + ".csv", "a", encoding="UTF-8"
) as file:
    writer = csv.writer(file)
    writer.writerow(genes)

def fitness_function(accuracy: float, density: float, depth: int = None) -> Union[np.float64, list[np.float64]]:
    fitness_score = accuracy + 0.05*density
    return fitness_score
backend_coupling_map = backend.coupling_map
# Torino coupling map should be: [[16, 23], [23, 16], [17, 27], [27, 17], [18, 31], [31, 18], [34, 40], [40, 34], [35, 44], [44, 35], [36, 48], [48, 36], [55, 65], [65, 55], [56, 69], [69, 56], [60, 61], [61, 60], [74, 86], [86, 74], [78, 79], [79, 78], [93, 103], [103, 93], [94, 107], [107, 94], [98, 99], [99, 98], [111, 120], [120, 111], [112, 124], [124, 112], [116, 117], [117, 116], [4, 16], [16, 4], [8, 17], [17, 8], [12, 18], [18, 12], [21, 34], [34, 21], [25, 35], [35, 25], [29, 36], [36, 29], [46, 55], [55, 46], [50, 56], [56, 50], [59, 60], [60, 59], [63, 73], [73, 63], [67, 74], [74, 67], [79, 80], [80, 79], [84, 93], [93, 84], [88, 94], [94, 88], [97, 98], [98, 97], [101, 111], [111, 101], [105, 112], [112, 105], [117, 118], [118, 117], [122, 131], [131, 122], [126, 132], [132, 126], [2, 3], [3, 2], [6, 7], [7, 6], [10, 11], [11, 10], [15, 19], [19, 15], [22, 23], [23, 22], [26, 27], [27, 26], [30, 31], [31, 30], [37, 52], [52, 37], [44, 45], [45, 44], [48, 49], [49, 48], [53, 57], [57, 53], [54, 61], [61, 54], [64, 65], [65, 64], [68, 69], [69, 68], [72, 78], [78, 72], [75, 90], [90, 75], [82, 83], [83, 82], [86, 87], [87, 86], [91, 95], [95, 91], [92, 99], [99, 92], [102, 103], [103, 102], [106, 107], [107, 106], [110, 116], [116, 110], [113, 128], [128, 113], [120, 121], [121, 120], [124, 125], [125, 124], [1, 2], [2, 1], [5, 6], [6, 5], [9, 10], [10, 9], [13, 14], [14, 13], [19, 20], [20, 19], [23, 24], [24, 23], [27, 28], [28, 27], [31, 32], [32, 31], [39, 40], [40, 39], [43, 44], [44, 43], [47, 48], [48, 47], [51, 52], [52, 51], [57, 58], [58, 57], [61, 62], [62, 61], [65, 66], [66, 65], [69, 70], [70, 69], [77, 78], [78, 77], [81, 82], [82, 81], [85, 86], [86, 85], [89, 90], [90, 89], [95, 96], [96, 95], [99, 100], [100, 99], [103, 104], [104, 103], [107, 108], [108, 107], [115, 116], [116, 115], [119, 120], [120, 119], [127, 128], [128, 127], [0, 15], [15, 0], [3, 4], [4, 3], [7, 8], [8, 7], [11, 12], [12, 11], [21, 22], [22, 21], [25, 26], [26, 25], [29, 30], [30, 29], [38, 53], [53, 38], [42, 54], [54, 42], [45, 46], [46, 45], [49, 50], [50, 49], [59, 72], [72, 59], [63, 64], [64, 63], [67, 68], [68, 67], [71, 75], [75, 71], [76, 91], [91, 76], [80, 92], [92, 80], [83, 84], [84, 83], [87, 88], [88, 87], [97, 110], [110, 97], [101, 102], [102, 101], [105, 106], [106, 105], [109, 113], [113, 109], [114, 129], [129, 114], [118, 130], [130, 118], [121, 122], [122, 121], [125, 126], [126, 125], [0, 1], [1, 0], [4, 5], [5, 4], [8, 9], [9, 8], [12, 13], [13, 12], [20, 21], [21, 20], [24, 25], [25, 24], [28, 29], [29, 28], [38, 39], [39, 38], [42, 43], [43, 42], [46, 47], [47, 46], [50, 51], [51, 50], [58, 59], [59, 58], [62, 63], [63, 62], [66, 67], [67, 66], [70, 71], [71, 70], [76, 77], [77, 76], [80, 81], [81, 80], [84, 85], [85, 84], [88, 89], [89, 88], [96, 97], [97, 96], [100, 101], [101, 100], [104, 105], [105, 104], [108, 109], [109, 108], [114, 115], [115, 114], [118, 119], [119, 118], [122, 123], [123, 122], [73, 82], [82, 73], [33, 37], [37, 33], [32, 33], [33, 32], [126, 127], [127, 126], [40, 41], [41, 40], [123, 124], [124, 123], [41, 42], [42, 41]]
# Or the analogous but as a list of tuples [(16, 23), (23, 16), ...]
backend_basis_gates = ['cz', 'id', 'sx', 'x', 'rz']
flattened_qsvm_connections = [
    item for sublist in qsvm_connections for item in sublist]
fmap, x_idxs = to_quantum(
    genes, gate_dict, nb_features, gates_per_qubits, nb_qubits, 0, qsvm_connections[0], coupling_map = backend_coupling_map
)

nb_samples = data_cv.shape[0]
generation_kernels = np.zeros((nb_samples, nb_samples, 21))
combined_circuits = []
max_circuit_per_job = 300 # depends on the backend
counter = 0

for i in range(nb_samples):
    print("preparing all QSVM circuits (this can take a few minutes and take up to 2 GB RAM)")
    for j in range(i+1, data_cv.shape[0]):
        combined_circuit = QuantumCircuit(132, nb_cbits)
        for k in range(21):
            bound_circuit = fmap.assign_parameters(data_cv[i, x_idxs]).compose(
                fmap.assign_parameters(data_cv[j, x_idxs]).inverse())
            combined_circuit.compose(bound_circuit, qubits=[
                                    qsvm_connections[k][l] for l in range(nb_qubits)], inplace=True)
        combined_circuit.measure(flattened_qsvm_connections, cbits)
        if counter % max_circuit_per_job == 0:
            combined_circuit_batch = []
        combined_circuit_batch.append(
            transpile(combined_circuit, basis_gates=backend_basis_gates, coupling_map=backend_coupling_map, optimization_level=0))

        if counter % max_circuit_per_job == max_circuit_per_job - 1 or counter == (nb_samples**2 - nb_samples)/2 - 1:            
            combined_circuits.append(combined_circuit_batch)
        counter += 1

# Prepare, send, retrieve jobs

job_results = []
print("Running jobs")
with Session(backend=backend): #, service=service):
    job_executions = [backend.run(combined_circuits[i], shots=8000) for i in range(counter // max_circuit_per_job + 1)]
for i in range(counter // max_circuit_per_job + 1):
    job_executions[i].wait_for_final_state()
    job_results.append(job_executions[i].result())

n_jobs= len(job_results)
big_counts = []
for i in range(n_jobs):
    print("getting counts from job", i)
    big_counts.append(job_results[i].get_counts())
counter = 0

print("Creating kernel matrices from job output. This can take a few minutes")
for i in range(nb_samples):
    for j in range(i+1, nb_samples):
        n_job = counter // max_circuit_per_job
        counts = big_counts[n_job][counter % max_circuit_per_job]
        counter += 1
        for k in range(21):
            generation_kernels[i, j, k] = get_kernel_entry(cbits[nb_qubits*k:nb_qubits*(k+1)], counts, nb_qubits)

# Symmetrizing and adding 1s to kernel diagonals
for k in range(21):
    generation_kernels[:, :, k] += generation_kernels[:,
                                                        :, k].T + np.eye(nb_samples)
    clf = SVC(kernel="precomputed")

    param_grid = {"C": [0.01, 0.1, 1, 10, 100, 1000, 10000]}
    grid_search = GridSearchCV(
        clf, param_grid, cv=5, scoring="accuracy", verbose=0)
    grid_search.fit(generation_kernels[:, :, k], data_labels)
    best_clf = grid_search.best_estimator_
    accuracy_cv_cost = cross_val_score(
        best_clf,
        generation_kernels[:, :, k],
        data_labels,
        cv=5,
        scoring="accuracy",
    ).mean()
    qker_matrix_0 = generation_kernels[:, :, k][data_labels == 0]
    qker_matrix_0 = np.triu(qker_matrix_0[:, data_labels == 0], 1)
    qker_array_0 = qker_matrix_0[np.triu_indices(
        qker_matrix_0.shape[0], 1)]
    qker_matrix_1 = generation_kernels[:, :, k][data_labels == 1]
    qker_matrix_1 = np.triu(qker_matrix_1[:, data_labels == 1], 1)
    qker_array_1 = qker_matrix_1[np.triu_indices(
        qker_matrix_1.shape[0], 1)]

    qker_matrix_01 = generation_kernels[:, :, k][data_labels == 0]
    qker_matrix_01 = qker_matrix_01[:, data_labels == 1]

    sparsity_cost = (np.mean(qker_array_0) + np.mean(qker_array_1)) / 2 - np.mean(
        qker_matrix_01
    )
    offdiagonal_mean = np.mean(np.triu(generation_kernels[:, :, k], 1))
    offdiagonal_std = np.std(np.triu(generation_kernels[:, :, k], 1))
    fitness_value = fitness_function(
        accuracy_cv_cost, offdiagonal_std)
    print("sparsity", sparsity_cost)
    print("accuracy", accuracy_cv_cost)
    print("fitness_value", fitness_value)

    with open(
        save_path + "/kernels_flattened" + suffix + ".csv", "a", encoding="UTF-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(generation_kernels[:, :, k].reshape(-1))
    with open(
        save_path + "/sparsity" + suffix + ".txt", "a", encoding="UTF-8"
    ) as file:
        file.write(str(sparsity_cost) + "\n")
    with open(
        save_path + "/accuracy" + suffix + ".txt", "a", encoding="UTF-8"
    ) as file:
        file.write(str(accuracy_cv_cost) + "\n")
    with open(
        save_path + "/fitness_values_iter_" + suffix + ".txt", "a", encoding="UTF-8"
    ) as file:
        file.write(str(fitness_value) + "\n")
    with open(
        save_path + "/offdiagonal_mean_" + suffix + ".txt", "a", encoding="UTF-8"
    ) as file:
        file.write(str(offdiagonal_mean) + "\n")
    with open(
        save_path + "/offdiagonal_std_" + suffix + ".txt", "a", encoding="UTF-8"
    ) as file:
        file.write(str(offdiagonal_std) + "\n")

# Saving last-generation information that will be loaded by the fitness function.
    with open(save_path + "/last_generation_fitness_values_" + suffix + ".csv", "w" if k == 0 else "a", encoding="UTF-8") as file:
        writer = csv.writer(file)
        writer.writerow(fitness_value) if hasattr(
            fitness_value, "len") > 1 else writer.writerow([fitness_value])
    with open(save_path + "/last_generation_genes_" + suffix + ".csv", "w" if k == 0 else "a", encoding="UTF-8") as file:
        writer = csv.writer(file)
        writer.writerow(genes)
