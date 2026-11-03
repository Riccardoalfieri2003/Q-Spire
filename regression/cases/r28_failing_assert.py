# expected: IM on qubit 0 (the function ends with a failing assertion that AutoFix cannot fix)
from qiskit import QuantumCircuit


def check():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    qc.x(0)
    assert qc.size() == 99
