# expected: nothing; qr2[1] is a qubit of the second register (index 3), measured last
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def regs():
    qr1 = QuantumRegister(2, "a")
    qr2 = QuantumRegister(2, "b")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr1, qr2, cr)
    qc.h(qr1[1])
    qc.cx(qr1[1], qr2[1])
    qc.measure(qr2[1], cr[0])
    return qc
