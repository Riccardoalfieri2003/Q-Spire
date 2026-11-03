# expected: nothing (a project's own transpile wrapper, not Qiskit's)
from qiskit import QuantumCircuit


def transpile(circuit, target):
    return circuit


def compile_it():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    out = transpile(qc, "qasm")
    print(out.size())
