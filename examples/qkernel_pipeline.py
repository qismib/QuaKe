# Importing libraries
from quake.utils.utils import load_runcard
from pathlib import Path
from quake.models.qsvm.qsvm_tester import get_features, make_kernels, SvmsComparison
from quake.models.qsvm import quantum_featuremaps
from qiskit.circuit import ParameterVector
from qiskit import Aer
from qiskit.utils import QuantumInstance

import numpy as np

data_folder = Path('../../output/tmp/data')
train_folder = Path('../../output/tmp/models/svm')
setup = load_runcard("../../output/tmp/cards/runcard.yaml")
dataset, labels = get_features(data_folder, train_folder, 'cnn', setup)

qubits = dataset[-1].shape[-1]
x = ParameterVector('x', length=qubits)
backend = Aer.get_backend('statevector_simulator')
#backend = Aer.get_backend('qasm_simulator')
quantum_instance = backend
#quantum_instance = QuantumInstance(backend, shots=8192, seed_simulator=42, seed_transpiler=42) # 8192

zf = quantum_featuremaps.z_featuremap(x, qubits,1)
zzf = quantum_featuremaps.zz_featuremap(x, qubits, 1)
c1 = quantum_featuremaps.custom1_featuremap(x, qubits, 1)
c2 = quantum_featuremaps.custom2_featuremap(x, qubits, 1)

quantum_kernels = make_kernels([zf, zzf, c1, c2], quantum_instance)

settings = {
    'x': x,
    'quantum_featuremaps': [zf, zzf, c1, c2],
    'quantum_instance': quantum_instance,
    'quantum_kernels': quantum_kernels,
    'kernel_names': ['Z', 'ZZ', 'Custom1', 'Custom2'],
    'cs': [10, 1000, 1000, 100],
    'training_size': [20, 30],
    'val_size': 100,
    'test_size': 100,
    'folder_name': Path('Statevector_1rep_highsample'),
    'folds': 2,
    'backend': backend,
} 

comparer = SvmsComparison(**settings)
dataset = [np.pi/2*dataset[0], np.pi/2*dataset[1], np.pi/2*dataset[2]]

comparer.train_svms_cv(dataset, labels)
comparer.plot_data(dataset, labels)
comparer.train_svms(dataset, labels)

comparer.learning_curves()

comparer.plot_decision_boundaries(cheap_version = True)
comparer.plot_bloch_spheres(dataset, labels)
comparer.plot_kernels()
comparer.plot_featuremaps()
comparer.save(settings)

# copy = SvmsComparison(**settings)
# copy.load_files(comparer.path)
# copy.plot_decision_boundaries()
# copy.save(settings)
