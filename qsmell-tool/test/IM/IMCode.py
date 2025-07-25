from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

qreg = QuantumRegister(3, 'q')
creg = ClassicalRegister(2, 'c')
qc = QuantumCircuit(qreg, creg)

qc.h(0)
qc.cx(0, 1)
qc.measure(0, creg[0])
qc.rz(1.57, 2)
qc.measure(1, creg[1])



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



"""
{
    0: [
        (0, {'operation_name': 'h', 'qubits_affected': [0], 'clbits_affected': [], 'row': 7, 'column_start': 0, 'column_end': 7, 'source_line': 'qc.h(0)'}), 
        (1, {'operation_name': 'cx', 'qubits_affected': [0, 1], 'clbits_affected': [], 'row': 8, 'column_start': 0, 'column_end': 11, 'source_line': 'qc.cx(0, 1)'}), 
        (2, {'operation_name': 'measure', 'qubits_affected': [0, 0], 'clbits_affected': [], 'row': 9, 'column_start': 0, 'column_end': 22, 'source_line': 'qc.measure(0, creg[0])'}), 
        (2, {'operation_name': 'measure', 'qubits_affected': [0, 0], 'clbits_affected': [], 'row': 9, 'column_start': 0, 'column_end': 22, 'source_line': 'qc.measure(0, creg[0])'}), 
        (4, {'operation_name': 'measure', 'qubits_affected': [0, 0], 'clbits_affected': [], 'row': 11, 'column_start': 0, 'column_end': 22, 'source_line': 'qc.measure(0, creg[0])'}), 
        (4, {'operation_name': 'measure', 'qubits_affected': [0, 0], 'clbits_affected': [], 'row': 11, 'column_start': 0, 'column_end': 22, 'source_line': 'qc.measure(0, creg[0])'})
    ], 
    
    1: [
        (1, {'operation_name': 'cx', 'qubits_affected': [0, 1], 'clbits_affected': [], 'row': 8, 'column_start': 0, 'column_end': 11, 'source_line': 'qc.cx(0, 1)'})
    ], 
    
    2: [
        (3, {'operation_name': 'rz', 'qubits_affected': [2], 'clbits_affected': [], 'row': 10, 'column_start': 0, 'column_end': 14, 'source_line': 'qc.rz(1.57, 2)', 'params': ['1.57']})
        ]
        
}
"""