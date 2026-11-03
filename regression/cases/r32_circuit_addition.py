# expected: IM on qubit 0 only (qc += oracle and qc1 + qc2 were removed in Qiskit 1.0; a failed
# += left two barriers next to each other, i.e., a false ROC)
from qiskit import QuantumCircuit


def grover_like():
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.barrier()
    oracle = QuantumCircuit(2)
    oracle.cz(0, 1)
    qc += oracle
    qc.barrier()
    qc.measure(0, 0)
    print(qc.size())


def joined():
    first = QuantumCircuit(1, 1)
    first.measure(0, 0)
    second = QuantumCircuit(1, 1)
    second.x(0)
    both = first + second
    print(both.size())
