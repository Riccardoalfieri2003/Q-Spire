from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister (3, 'q')
creg_c = ClassicalRegister (3, 'c')
qc = QuantumCircuit (qreg_q, creg_c)
#qc.h (qreg_q)
#qc.h(qreg_q[0])
qc.p(pi / 2, qreg_q[0])
qc.z(qreg_q[0])
#qc.s(qreg_q[1])
qc.s(qreg_q[0])
#qc.measure (qreg_q[0], creg_c[0] )
qc.barrier ()
#qc.h(qreg_q[1])
qc.p(pi / 4, qreg_q[1])
qc.z(qreg_q[1])
qc.s(qreg_q[1])
#qc.measure (qreg_q[1], creg_c[1])
qc.barrier ()
qc.h(qreg_q[2])
qc.p(pi / 8, qreg_q[2])
qc.z(qreg_q[2])
qc.s(qreg_q[2])

qc.measure_all(add_bits=False) #Da rimuovere per non avere smell
#qc.measure (qreg_q[2], creg_c[2])




"""qc=QuantumCircuit(3)
qc.h(0)
qc.z(1)
qc.cx(0,1)
qc.id(0)
qc.h(2)
qc.cx(1,2)
qc.z(1)
qc.z(1)
qc.h(2)
"""

import matplotlib.pyplot as plt

fig = qc.draw(output="mpl")
plt.show()