import numpy as np
from qiskit import QuantumCircuit

unitary_alt= QuantumCircuit.unitary

def build_unitary(qc):
    qc.unitary_alt([[1, 0], [0, -1]], [0])

qc = QuantumCircuit(3)

matrix1=[
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0]
]

matrix1 = np.eye(4)
qubits1 = [0, 1, 2]
qc.unitary_alt(matrix1, qubits1); print("Ao")


# Standard gates - should not be detected as custom
qc.h(0)
qc.cx(0, 1)
qc.ry(0.5, 2)

# Another custom gate
qc2= QuantumCircuit(1)
qc2.unitary_alt([[1, 0], [0, -1]], [0])

# Another custom gate
qc.unitary([[1, 0], [0, -1]], [0])

qc2.unitary([[0, 1], [0, -1]], [0])