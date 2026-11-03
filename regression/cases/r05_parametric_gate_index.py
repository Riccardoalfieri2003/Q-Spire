# expected: nothing; params[2] is a parameter index, not qubit 2
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def ansatz():
    params = ParameterVector("p", 3)
    qc = QuantumCircuit(1)
    qc.rx(params[0], 0)
    qc.ry(params[1], 0)
    qc.rz(params[2], 0)
    return qc
