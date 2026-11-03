# expected: LC (l*c >= 20: l=10, c=4) and ROC (identical consecutive layers)
from qiskit import QuantumCircuit


def deep(n=4, reps=10):
    qc = QuantumCircuit(n)
    for _ in range(reps):
        for q in range(n):
            qc.h(q)
    return qc
