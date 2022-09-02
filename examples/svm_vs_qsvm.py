"""This module starts a training session and return a comparison between SVM and QSVM performance
"""
from qiskit.providers.aer import AerSimulator
from quake.utils.utils import load_runcard
from pathlib import Path
from quake.models.qsvm.qsvm_tester import get_features, make_kernels, SvmsComparison
from quake.models.qsvm import quantum_featuremaps
from qiskit.circuit import ParameterVector
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit import IBMQ

# Loading the dataset
input_folder = Path("../../output")
setup = load_runcard("../../output/cards/runcard.yaml")
output_folder = Path("CNN_results/Statevector_1rep_Bal_ds_10feats")
dataset, labels = get_features(input_folder, "cnn", setup)

# Defining the quantum circuits
qubits = dataset[-1].shape[-1]
x = ParameterVector("x", length=qubits)

# Statevector backend
backend = AerSimulator(method="statevector")

# Qasm backend
alternative_backend = Aer.get_backend("qasm_simulator")
alternative_instance = QuantumInstance(
    alternative_backend, shots=8192, seed_simulator=42, seed_transpiler=42
)

# Simulated IBM's Lagos backend
IBMQ.load_account()
real_hw = IBMQ.get_provider("ibm-q-research-2").backends("ibm_lagos")[0]
noisy_backend = AerSimulator.from_backend(real_hw)
noisy_instance = QuantumInstance(
    noisy_backend, shots=8192, seed_simulator=42, seed_transpiler=42
)

# Loading some quantum featuremaps
zf = quantum_featuremaps.z_featuremap(x, qubits, 1, align=False)
zzf = quantum_featuremaps.zz_featuremap(x, qubits, 1, align=False)
c1 = quantum_featuremaps.custom1_featuremap(x, qubits, 1, align=False)
c2 = quantum_featuremaps.custom2_featuremap(x, qubits, 1, align=False)
gen1 = quantum_featuremaps.genetic_featuremap(x, 1, align=False)
gen2 = quantum_featuremaps.genetic_featuremap_2(x, 1, align=False)
genatt = quantum_featuremaps.genetic_attention(x, 1, align=False)
# Creating the kernel list
quantum_kernels = make_kernels([zf, zzf, c1, c2], backend)

# Defining hyperparameters for classical SVMs
linear = {"kernel": "linear", "C": 100, "gamma": 0.01}
poly = {"kernel": "poly", "C": 10, "degree": 2}
rbf = {"kernel": "rbf", "C": 10, "gamma": 0.1}
opts = [linear, poly, rbf]

# Settings dictionary
settings = {
    "x": x,
    "quantum_featuremaps": [zf, zzf, c1, c2],
    "quantum_kernels": quantum_kernels,
    "kernel_names": ["Z", "ZZ", "c1", "c2"],
    "cs": [100, 100, 100, 100],
    "training_size": [10, 20, 50, 100, 200, 500, 1000, 2000],
    "val_size": 300,
    "test_size": 300,
    "folder_name": output_folder,
    "folds": 20,
    "backend": backend,
    "classic_opts": opts,
}

comparer = SvmsComparison(**settings)

# comparer.compare_backend(dataset, labels, alternative_backend) # Uncomment for comparing different backends
comparer.train_svms_cv(dataset, labels)
comparer.plot_data(dataset, labels)
comparer.train_svms(dataset, labels)
comparer.learning_curves()
comparer.plot_bloch_spheres(dataset, labels, with_prediction=False)
comparer.plot_bloch_spheres(dataset, labels, with_prediction=True)
comparer.plot_kernels()
comparer.plot_featuremaps()
comparer.save(settings)

# Loading data from previously trained sessions
# copy_comparer = SvmsComparison(**settings)
# copy_comparer.load_files(comparer.path)
