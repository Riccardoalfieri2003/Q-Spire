# expected: CG (QuantumCircuit.unitary with an explicit matrix)
import numpy as np
from qiskit import QuantumCircuit


def custom():
    qc = QuantumCircuit(1)
    m = np.array([[0, 1], [1, 0]])
    qc.unitary(m, [0], label="myx")
    print(qc.size())
