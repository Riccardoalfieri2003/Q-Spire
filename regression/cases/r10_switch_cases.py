# expected: IM on qubit 0 (measure followed by the switch), no ROC (exclusive cases are not consecutive)
from qiskit import QuantumCircuit


def switch():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    with qc.switch(qc.clbits[0]) as case:
        with case(0):
            qc.x(0)
        with case(1):
            qc.x(0)
    return qc
