# expected: nothing (only terminal measurements, no repetition)
from qiskit import QuantumCircuit


def terminal():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
