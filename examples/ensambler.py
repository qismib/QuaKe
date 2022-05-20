from pathlib import Path
from quake.models.qsvm.qsvm_tester import get_features, make_kernels, SvmsComparison
import numpy as np
from quake.models.qsvm import quantum_featuremaps
from qiskit.circuit import ParameterVector
from qiskit import Aer
from qiskit.utils import QuantumInstance
qubits = 2
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
    'training_size': [20, 30, 50, 100, 150],
    'val_size': 300,
    'test_size': 300,
    'folder_name': Path('Statevector_1rep'),
    'folds': 20,
    'backend': backend,
} 
path = Path('../../Statevector_1rep')
copy = SvmsComparison(**settings)
copy.load_files(path)

dataset = copy.validation[-1][0][0]
labels = copy.validation[-1][0][1]
predsum = np.zeros(labels.shape)


val_fold0 = copy.validation_preds[-1][0]
for i in range(len(predsum)):
    for prediction in val_fold0:
        predsum[i] = predsum[i] + prediction[i]
    predsum[i] = predsum[i]/len(val_fold0) #/len(val_150_fold)

cs = np.arange(0, 1, 0.05)

acc = np.zeros(cs.shape)
for i, c in enumerate(cs):
    y = predsum > c
    acc[i] = np.sum(y == labels)/len(labels)
import pdb; pdb.set_trace()
