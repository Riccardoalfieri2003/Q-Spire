# expected: nothing (a measurement followed only by a barrier is terminal)
from qiskit import QuantumCircuit


def draw_friendly():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.barrier()
    print(qc.size())
