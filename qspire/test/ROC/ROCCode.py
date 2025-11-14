from qiskit import QuantumCircuit

qc = QuantumCircuit (3, 3) # 3 Quantum and 3 Classical registers

#measureQubit = QuantumCircuit (1, 1, name='M' )
#measureQubit.measure(0, 0)

#hadamard = QuantumCircuit (1, name=' H' )
#hadamard.h(0)

for i in range (3):
    for j in range (3):
        #qc.append (hadamard, [j])
        qc.h([j])
    for j in range (3) :
        #qc.append (measureQubit, [j], [j])
        qc.measure([j],[j])
    qc.barrier ()

"""for j in range(9):
    hadamard.h(0)"""

qc.repeat(3)
#qc.h(1)