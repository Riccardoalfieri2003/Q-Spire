# expected: nothing (u2 then u3 are different gates; mapping both to `u` created a repetition)
from qiskit import QuantumCircuit


def old_gates():
    qc = QuantumCircuit(1)
    qc.u2(0.1, 0.2, 0)
    qc.u3(0.1, 0.2, 0.3, 0)
    print(qc.size())
