# expected (static mode): nothing, module-level code is not analyzed; (dynamic mode): IM on qubit 0
from qiskit import QuantumCircuit

qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)
qc.x(0)
