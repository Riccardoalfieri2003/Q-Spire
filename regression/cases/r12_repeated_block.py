# expected: ROC (the same two-gate block is repeated consecutively)
from qiskit import QuantumCircuit


def repeated():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(0, 1)
    return qc
