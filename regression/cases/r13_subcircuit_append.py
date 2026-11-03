# expected: IM on qubit 0 inside an appended sub-circuit (measure then x on qubit 0)
from qiskit import QuantumCircuit


def outer():
    sub = QuantumCircuit(1, 1)
    sub.measure(0, 0)
    sub.x(0)
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.append(sub.to_instruction(), [0], [0])
    return qc
