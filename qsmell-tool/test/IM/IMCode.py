from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

qreg = QuantumRegister(3, 'q')
creg = ClassicalRegister(1, 'c')
qc = QuantumCircuit(qreg, creg)

qc.h(0)
qc.cx(0, 1)
qc.measure(0, creg[0])
qc.rz(1.57, 2)
qc.measure(0, creg[0])



qreg_q=QuantumRegister(3,'q')
creg_c=ClassicalRegister(2,'c')
qc2=QuantumCircuit(qreg_q,creg_c)


qc2.h(qreg_q[0])

qc2.z(1)
qc2.h(qreg_q[0])
qc2.cx(0,1)
qc2.measure(qreg_q[0],creg_c[0])
#qc2.measure(qreg_q[0],creg_c[1])
qc2.measure(qreg_q[0],creg_c[1])
