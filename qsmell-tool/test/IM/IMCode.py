from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

qreg_q = QuantumRegister (3, 'q' )
creg_c = ClassicalRegister (2, 'c' )
qc = QuantumCircuit (qreg_q, creg_c)

qc.h(qreg_q[0])
qc.measure (qreg_q[0], creg_c[0])
# qc.cnot (qreg_q[0], qreg_q[1])
# qc.cnot (qreg_q[0], qreg_q[2])
qc.h(qreg_q[0])
qc.measure (qreg_q[0], creg_c [1] )
# qc.measure (qreg_q[1], creg_c[0] )
# qc.measure (qreg_q[2], creg_c[1])