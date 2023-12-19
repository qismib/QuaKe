import numpy as np
import matplotlib.pyplot as plt
from quake.models.qsvm import genetic_main as genetic
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict

gen0 = np.array(
    [
        9,
        1,
        1,
        14,
        6,
        2,
        0,
        0,
        3,
        1,
        1,
        1,
        1,
        10,
        4,
        0,
        1,
        0,
        7,
        2,
        4,
        0,
        1,
        0,
        7,
        4,
        1,
        0,
        10,
        2,
        6,
        1,
        1,
        6,
        12,
        8,
        1,
        0,
        11,
        7,
        8,
        1,
        0,
        13,
        3,
        0,
        1,
        1,
        17,
        4,
        8,
        1,
        1,
        13,
        4,
        7,
        1,
        1,
        0,
        4,
    ]
)

NB_QUBITS = 3
GATES_PER_QUBITS = 3
gate_dict = OrderedDict(
    [
        ("single_non_parametric", ["I", "H", "X"]),
        ("single_parametric", ["RX", "RY", "RZ"]),
        ("two_non_parametric", ["CX"]),
        ("two_parametric", ["CRX", "CRY", "CRZ", "CP"]),
    ]
)
nb_features = 18
fmaps, _ = genetic.to_quantum(gen0, gate_dict, nb_features, GATES_PER_QUBITS, NB_QUBITS)
import pdb

pdb.set_trace()
