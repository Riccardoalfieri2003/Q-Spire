# expected: IM on qubit 0 (once)
from qiskit import QuantumCircuit


def build():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    qc.x(0)
