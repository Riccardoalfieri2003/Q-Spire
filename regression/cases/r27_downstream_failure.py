# expected: IM on qubit 0 (later code fails on the mock circuit; AutoFix used to delete the circuit)
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag


def check():
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    qc.x(0)
    dag = circuit_to_dag(qc)
    assert dag.depth() == 2
