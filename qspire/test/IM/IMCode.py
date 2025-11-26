
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit 


def have_smell(qc:QuantumCircuit): # type: ignore
    qreg_q = QuantumRegister (3, 'q' ) 
    creg_c = ClassicalRegister (2, 'c' )
    qc = QuantumCircuit (qreg_q, creg_c) 

    qc.h(qreg_q[0]) 
    qc.measure (qreg_q[0], creg_c[0] ) 
    qc.h(qreg_q[0]) 
    qc.measure (qreg_q[0], creg_c[1])