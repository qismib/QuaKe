from qiskit.circuit import QuantumCircuit
import numpy as np
from qiskit.circuit import ParameterVector


def z_featuremap(x, qubits, repeats):
    z_featuremap = QuantumCircuit(qubits)
    # theta = ParameterVector("θ", qubits)
    # for q in range(qubits):
    #     z_featuremap.ry(theta[q], q)
    for i in range(repeats):
        for q in range(qubits):
            z_featuremap.h(q)
            z_featuremap.p(2*x[q],q)
        z_featuremap.barrier()
    return z_featuremap 


def zz_featuremap(x, qubits, repeats):
    zz_featuremap = QuantumCircuit(qubits)
    # theta = ParameterVector("θ", qubits)
    # for q in range(qubits):
    #     zz_featuremap.ry(theta[q], q)
    for i in range(repeats):
        for q in range(qubits):
            zz_featuremap.h(q)
            zz_featuremap.p(2*x[q], q)
        for q in range(qubits-1):
            zz_featuremap.cx(q, q+1)
            zz_featuremap.p(2*(np.pi-x[q])*(np.pi-x[q+1]), q+1)
            zz_featuremap.cx(q, q+1)

        zz_featuremap.barrier()
    return zz_featuremap  


def custom1_featuremap(x, qubits, repeats):
    custom1_featuremap = QuantumCircuit(qubits)
    # theta = ParameterVector("θ", qubits)
    # for q in range(qubits):
    #     custom1_featuremap.ry(theta[q], q)
    for _ in range(repeats):
        for q in range(qubits):
            if q % 2 == 0: 
                custom1_featuremap.ry(2*x[q], q)
                custom1_featuremap.p(2*x[q], q)
                #custom1_featuremap.h(q)
            else:
                custom1_featuremap.rx(2*x[q], q)
                custom1_featuremap.p(2*x[q], q)
        for i in range(qubits-1):
            #for j in range(i+1, qubits):
            custom1_featuremap.cx(i, i+1)
            custom1_featuremap.p(x[i] * x[i+1], i+1)
            custom1_featuremap.cx(i, i+1)
        custom1_featuremap.barrier()
    return custom1_featuremap  

def custom2_featuremap(x, qubits, repeats):
    custom2_featuremap = QuantumCircuit(qubits)
    # theta = ParameterVector("θ", qubits)
    # for q in range(qubits):
    #     custom2_featuremap.ry(theta[q], q)
    for _ in range(repeats):
        for q in range(qubits):
            if q % 2 == 0:
                custom2_featuremap.rx(np.arcsin(2/np.pi*x[q]), q)
                custom2_featuremap.rz(np.arccos(2/np.pi*2/np.pi*x[q]*x[q]), q)
            else:
                custom2_featuremap.rx(np.arcsin(2/np.pi*x[q]), q)
                custom2_featuremap.rz(np.arccos(2/np.pi*2/np.pi*x[q]*x[q]), q)
        custom2_featuremap.barrier()
    return custom2_featuremap 

def custom3_featuremap(x, qubits, repeats):
    custom1_featuremap = QuantumCircuit(qubits)
    theta = ParameterVector("θ", 1)
    for _ in range(repeats):
        for q in range(qubits):
            if q % 2 == 0: 
                custom1_featuremap.h(q)
                custom1_featuremap.rx(theta[0]*x[q], q)
            else:
                custom1_featuremap.ry(theta[0]*x[q], q)
        for i in range(qubits):
            for j in range(i+1, qubits):
                custom1_featuremap.cx(i, j)
                custom1_featuremap.p(np.sin(x[i]) * np.cos(x[j]), j)
                custom1_featuremap.cx(i, j)
        custom1_featuremap.barrier()
    return custom1_featuremap  