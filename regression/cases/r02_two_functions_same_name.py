# expected: nothing (two different circuits that share the variable name qc)
from qiskit import QuantumCircuit


def first():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.measure(0, 0)
    qc.measure(1, 1)


def second():
    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.x(1)
    qc.measure(0, 0)
    qc.measure(1, 1)
