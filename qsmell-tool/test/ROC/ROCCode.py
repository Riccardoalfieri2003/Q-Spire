from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

phi = Parameter('phi')
 


qc = QuantumCircuit (3, 3) # 3 Quantum and 3 Classical registers

z=QuantumCircuit (1, name=' Z' )
z.z(0)
z.h(0)
"""
z.z(0)
"""

'''
s
'''


#hadamard.h(0)
#hadamard.z(0)


measureQubit = QuantumCircuit (1, 1, name='M' )
measureQubit.measure(0, 0)

hadamard = QuantumCircuit (1, name=' H' )

for j in range (3) : 
     qc.append( hadamard, [j] )
     qc.z(0)
     hadamard.rx(phi*j, 0)
     qc.append(measureQubit, [j], [j])

for j in range (3): 
    qc.z([j])

qc.barrier ()

qc.repeat(3)



