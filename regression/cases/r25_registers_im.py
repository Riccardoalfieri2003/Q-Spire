# expected: IM on qubit 3 (qr2[1] is the fourth qubit of the circuit)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def regs():
    qr1 = QuantumRegister(2, "a")
    qr2 = QuantumRegister(2, "b")
    cr = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qr1, qr2, cr)
    qc.h(qr1[1])
    qc.cx(qr1[1], qr2[1])
    qc.measure(qr2[1], cr[0])
    qc.x(qr2[1])
    qc.measure(qr1, cr)
    return qc
