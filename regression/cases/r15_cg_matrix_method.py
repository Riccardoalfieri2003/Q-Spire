# expected: nothing (a .unitary() method that returns a matrix is not a custom gate, as in Cirq)
import numpy as np
from qiskit import QuantumCircuit


class ToyCircuit:
    def unitary(self):
        return np.eye(2)


def compare():
    a = ToyCircuit()
    b = ToyCircuit()
    ok = np.allclose(a.unitary(), b.unitary())
    qc = QuantumCircuit(1)
    qc.h(0)
    print(ok)
