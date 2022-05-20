#-------------
# noisy_backend = AerSimulator.from_backend(real_hw)
# real_hw = IBMQ.get_provider().backends(‘nome_device')[0]

#-------------
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
    'quantum_featuremaps': [zf, c2],
    'quantum_instance': quantum_instance,
    'quantum_kernels': [quantum_kernels[0],quantum_kernels[3]] ,
    'kernel_names': ['Custom1', 'Custom2'],
    'cs': [10, 1000, 1000, 100],
    'training_size': [2000],
    'val_size': 800,
    'test_size': 800,
    'folder_name': Path('Output_Folder_weird_feat_layer_altric'),
    'folds': 1,
    'backend': backend,
} 

comparer = SvmsComparison(**settings)

# comparer.train_svms_cv(dataset, labels)
comparer.plot_data(dataset, labels)
#comparer.train_svms(dataset, labels)

#comparer.learning_curves()

# comparer.plot_bloch_spheres(dataset, labels)
# comparer.plot_kernels()
# comparer.plot_featuremaps()
comparer.save(settings)
# comparer.plot_decision_boundaries(cheap_version = False)

# copy = SvmsComparison(**settings)
# copy.load_files(comparer.path)
# copy.plot_decision_boundaries()
# copy.save(settings)
