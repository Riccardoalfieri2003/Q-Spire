from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

phi = Parameter('phi')
 


qc = QuantumCircuit (3, 3) # 3 Quantum and 3 Classical registers

"""
z=QuantumCircuit (1, name=' Z' )
z.z(0)
z.h(0)
z.z(0)
"""




#hadamard.h(0)
#hadamard.z(0)




measureQubit = QuantumCircuit (1, 1, name='M' )
measureQubit.measure(0, 0)

"""for i in range(3):
    for j in range (3) : 
        qc.append( hadamard, [j] )
        qc.z(0)
        #qc.measure(0,0)
        qc.append(measureQubit, [j], [j])"""

hadamard = QuantumCircuit (1, name=' H' )
hadamard.h(0)
#hadamard.rx(phi, 0)

for i in range (3):
    for j in range (3):
        qc.append (hadamard, [j])
    for j in range (3) :
        qc.append (measureQubit, [j], [j])
    qc.barrier ()
        

"""for k in range(3):
    for i in range(3):
        for j in range (3): 
            qc.z([j])"""

#print(qc.data)

#qc.barrier ()

#qc.repeat(3)


