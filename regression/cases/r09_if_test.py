# expected: IM on qubit 0 (measure followed by a conditional operation on qubit 0)
from qiskit import QuantumCircuit


def dynamic():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(0)
    return qc
