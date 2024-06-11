"""We execute 21 identical qsvms to study site performance"""
# oldest job id with gridsearch:
# '' without gridsearch:  
import numpy as np
from pathlib import Path
from qiskit_ibm_runtime import QiskitRuntimeService, Session

from collections import OrderedDict
from quake.models.qsvm import genetic_v4 as genetic
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Callable, Union
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.compiler import transpile
from quake.models.qsvm.genetic_v4 import to_quantum, get_kernel_entry
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import csv
import time
import shutil
import os
import gc
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_provider import IBMProvider
# Save your credentials on disk.
# IBMProvider.save_account(token='468dbb546acdc519d5999381a9d30fc849d558af57ca14fb9c9572ce2f15b387f7a16cca2fad386252168a5a63a55fecc4e75637e6b2e7da9beb83bab5ec463d', overwrite=True)
# #ADD WITH SESSION    
# provider = IBMProvider(instance='ibm-q-cern/infn/qcnphepgw')
# backend = provider.get_backend('ibm_nazca')
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
    [126,125,124,123]]
gate_dict = OrderedDict(
    [
        ("single_non_parametric", ["Id", "X","SX"]),
        ("single_parametric", ["RZ", "RZ"]),
        ("two_non_parametric", ["ECR"]),
        ("two_parametric", []),
    ]
)

nb_qubits = 4
gates_per_qubits = 8
data_cv, data_labels = np.load("C:/Users/pc/Desktop/work/data_cv.npy"), np.load("C:/Users/pc/Desktop/work/data_labels.npy")
useful_ft = [0, 5, 6, 8, 10, 12, 14, 15]
data_cv = data_cv[:, useful_ft]

nb_features = data_cv.shape[1]
nb_cbits = 4*21
cbits = [item for item in range(0, nb_cbits)]
nb_samples = 100
############# CHOOSE BEST KERNEL FOUND IN RUN ###############
generation_zero = genetic.initial_population(
    1, nb_qubits, gates_per_qubits, gate_dict, nb_features
)
# import pdb; pdb.set_trace()
genes = generation_zero[0]
#############################################################
suffix = 'best_run'
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
backend_basis_gates = backend.basis_gates
flattened_qsvm_connections = [
    item for sublist in qsvm_connections for item in sublist]
fmap, x_idxs = to_quantum(
    genes, gate_dict, nb_features, gates_per_qubits, nb_qubits, 0, qsvm_connections[0], coupling_map = backend_coupling_map
)
print(fmap)

nb_samples = data_cv.shape[0]
generation_kernels = np.zeros((nb_samples, nb_samples, 21))
combined_circuits = []
max_circuit_per_job = 300 # depends on the backend
counter = 0

for i in range(nb_samples):
    print(i)
    for j in range(i+1, data_cv.shape[0]):
        combined_circuit = QuantumCircuit(127, nb_cbits)
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
print("Running job")
with Session(backend=backend): #, service=service):
    job_executions = [backend.run(combined_circuits[i], shots=8000) for i in range(counter // max_circuit_per_job + 1)]
for i in range(counter // max_circuit_per_job + 1):
    job_executions[i].wait_for_final_state()
    job_results.append(job_executions[i].result())

# print("Recovery mode. Fetching the last 17 job results")
# service = QiskitRuntimeService()
# job_results = []
# retrieved_job = service.jobs(backend_name="ibm_nazca", limit = 33)
# for i in range(17):
#     print(retrieved_job[-i - 1].job_id())
#     job_results.append(retrieved_job[-i - 1].result())

n_jobs= len(job_results)
big_counts = []
for i in range(n_jobs):
    print("getting counts from job", i)
    big_counts.append(job_results[i].get_counts())
counter = 0
for i in range(nb_samples):
    for j in range(i+1, nb_samples):
        print("now doing", counter)
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
