from qiskit.providers.aer import AerSimulator
from quake.utils.utils import load_runcard
from pathlib import Path
from quake.models.qsvm.qsvm_tester import get_features, make_kernels, SvmsComparison
from quake.models.qsvm import quantum_featuremaps
from qiskit.circuit import ParameterVector
from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit import IBMQ

input_folder = Path("../../output/tmp/")
setup = load_runcard("../../output/tmp/cards/runcard.yaml")
output_folder = Path("../../Output_QSVM")

dataset, labels = get_features(input_folder, "cnn", setup)

qubits = dataset[-1].shape[-1]
x = ParameterVector("x", length=qubits)
backend = AerSimulator(method="statevector")

alternative_backend = Aer.get_backend("qasm_simulator")
alternative_instance = QuantumInstance(
    alternative_backend, shots=8192, seed_simulator=42, seed_transpiler=42
)

# IBMQ.load_account()
# real_hw = IBMQ.get_provider('ibm-q-research-2').backends('ibm_lagos')[0]
# noisy_backend = AerSimulator.from_backend(real_hw)
# noisy_instance = QuantumInstance(noisy_backend, shots=8192, seed_simulator=42, seed_transpiler=42) # 8192

zf = quantum_featuremaps.z_featuremap(x, qubits, 1, align = False)
zzf = quantum_featuremaps.zz_featuremap(x, qubits, 1, align = False)
c1 = quantum_featuremaps.custom1_featuremap(x, qubits, 1, align = False)
c2 = quantum_featuremaps.custom2_featuremap(x, qubits, 1, align = False)

quantum_kernels = make_kernels([zf, zzf, c1, c2], backend)

linear = {"kernel": "linear", "C": 1, "gamma": 10}
poly = {"kernel": "poly", "C": 1, "degree": 3}
rbf = {"kernel": "rbf", "C": 10, "gamma": 1}
opts = [linear, poly, rbf]

settings = {
    "x": x,
    "quantum_featuremaps": [c2],
    "quantum_kernels": [quantum_kernels[3]],
    "kernel_names": ["Z", "ZZ", "c1", "c2"],
    "cs": [1000, 1000, 200, 1000],
    "training_size": [10, 20, 50, 100, 200, 500],
    "val_size": 800,
    "test_size": 1,
    "folder_name": output_folder,
    "folds": 20,
    "backend": backend,
    "classic_opts": opts,
}

comparer = SvmsComparison(**settings)
# comparer.compare_backend(dataset, labels, alternative_backend)

comparer.train_svms_cv(dataset, labels)
comparer.plot_data(dataset, labels)
comparer.train_svms(dataset, labels)
comparer.learning_curves()
comparer.plot_bloch_spheres(dataset, labels)
comparer.plot_kernels()
comparer.plot_featuremaps()
comparer.save(settings)
# comparer.plot_decision_boundaries(cheap_version = False)
