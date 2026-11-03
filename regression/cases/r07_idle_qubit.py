# expected: IdQ on qubit 1 (idle for more than two layers), IQ on qubit 1
from qiskit import QuantumCircuit


def idle():
    qc = QuantumCircuit(2)
    qc.h(1)
    qc.h(0)
    qc.x(0)
    qc.y(0)
    qc.z(0)
    qc.cx(0, 1)
    return qc
