# expected: nothing; delay(200, 0) acts on qubit 0, not qubit 200
from qiskit import QuantumCircuit


def scheduled():
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.delay(200, 0)
    qc.cx(0, 1)
    return qc
