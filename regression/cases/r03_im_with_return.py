# expected: IM on qubit 0
from qiskit import QuantumCircuit


def build():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(0)
    return qc
