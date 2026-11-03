# expected: nothing (a local function named unitary, as in PennyLane templates)
import numpy as np
from qiskit import QuantumCircuit


def unitary(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def use():
    m = unitary(0.3)
    qc = QuantumCircuit(1)
    qc.h(0)
    print(m.shape)
