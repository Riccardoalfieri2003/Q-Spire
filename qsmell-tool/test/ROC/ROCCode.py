from qiskit import QuantumCircuit

qc = QuantumCircuit (3, 3) # 3 Quantum and 3 Classical registers

hadamard = QuantumCircuit (1, name=' H' )
hadamard.h (0)

measureQubit = QuantumCircuit (1, 1, name='M' )
measureQubit.measure(0, 0)

z=QuantumCircuit (1, name=' Z' )
z.z(0)

for j in range (3) :
    qc.append (hadamard, [j])
for j in range (3):
    qc.append (measureQubit, [j], [j])
    qc.append(z, [j])
qc.barrier ()

qc.repeat(3)

"""


{
    'qc': [
        {'operation_name': ' H', 'qubits_affected': [0], 'clbits_affected': [], 'row': 15, 'column_start': 4, 'column_end': 29, 'source_line': 'qc.append (hadamard, [j])'}, 
        {'operation_name': ' H', 'qubits_affected': [1], 'clbits_affected': [], 'row': 15, 'column_start': 4, 'column_end': 29, 'source_line': 'qc.append (hadamard, [j])'}, 
        {'operation_name': ' H', 'qubits_affected': [2], 'clbits_affected': [], 'row': 15, 'column_start': 4, 'column_end': 29, 'source_line': 'qc.append (hadamard, [j])'}, 
        {'operation_name': 'M', 'qubits_affected': [0], 'clbits_affected': [0], 'row': 17, 'column_start': 4, 'column_end': 38, 'source_line': 'qc.append (measureQubit, [j], [j])'}, 
        {'operation_name': ' Z', 'qubits_affected': [0], 'clbits_affected': [], 'row': 15, 'column_start': 4, 'column_end': 29, 'source_line': 'qc.append (hadamard, [j])'}, 
        {'operation_name': 'M', 'qubits_affected': [1], 'clbits_affected': [1], 'row': 17, 'column_start': 4, 'column_end': 38, 'source_line': 'qc.append (measureQubit, [j], [j])'}, 
        {'operation_name': ' Z', 'qubits_affected': [1], 'clbits_affected': [], 'row': 15, 'column_start': 4, 'column_end': 29, 'source_line': 'qc.append (hadamard, [j])'}, 
        {'operation_name': 'M', 'qubits_affected': [2], 'clbits_affected': [2], 'row': 17, 'column_start': 4, 'column_end': 38, 'source_line': 'qc.append (measureQubit, [j], [j])'}, 
        {'operation_name': ' Z', 'qubits_affected': [2], 'clbits_affected': [], 'row': 15, 'column_start': 4, 'column_end': 29, 'source_line': 'qc.append (hadamard, [j])'}, 
        {'operation_name': 'barrier', 'qubits_affected': [0, 1, 2], 'clbits_affected': [], 'row': 19, 'column_start': 0, 'column_end': 13, 'source_line': 'qc.barrier ()'}
    ], 
    
    'hadamard': [
        {'operation_name': 'h', 'qubits_affected': [0], 'clbits_affected': [], 'row': 6, 'column_start': 0, 'column_end': 14, 'source_line': 'hadamard.h (0)'}
    ], 
    
    'measureQubit': [
        {'operation_name': 'measure', 'qubits_affected': [0], 'clbits_affected': [0], 'row': 9, 'column_start': 0, 'column_end': 26, 'source_line': 'measureQubit.measure(0, 0)'}
    ], 
    
    'z': [
        {'operation_name': 'z', 'qubits_affected': [0], 'clbits_affected': [], 'row': 12, 'column_start': 0, 'column_end': 6, 'source_line': 'z.z(0)'}], 'QuantumCircuit': []
    }


"""

#print(qc.data)

"""
[
    CircuitInstruction(operation=Instruction(name=' H', num_qubits=1, num_clbits=0, params=[]), qubits=(<Qubit register=(3, "q"), index=0>,), clbits=()), 
    CircuitInstruction(operation=Instruction(name=' H', num_qubits=1, num_clbits=0, params=[]), qubits=(<Qubit register=(3, "q"), index=1>,), clbits=()), 
    CircuitInstruction(operation=Instruction(name=' H', num_qubits=1, num_clbits=0, params=[]), qubits=(<Qubit register=(3, "q"), index=2>,), clbits=()),
    CircuitInstruction(operation=Instruction(name='M', num_qubits=1, num_clbits=1, params=[]), qubits=(<Qubit register=(3, "q"), index=0>,), clbits=(<Clbit register=(3, "c"), index=0>,)), 
    CircuitInstruction(operation=Instruction(name='M', num_qubits=1, num_clbits=1, params=[]), qubits=(<Qubit register=(3, "q"), index=1>,), clbits=(<Clbit register=(3, "c"), index=1>,)), 
    CircuitInstruction(operation=Instruction(name='M', num_qubits=1, num_clbits=1, params=[]), qubits=(<Qubit register=(3, "q"), index=2>,), clbits=(<Clbit register=(3, "c"), index=2>,)), 
    CircuitInstruction(operation=Instruction(name='barrier', num_qubits=3, num_clbits=0, params=[]), qubits=(<Qubit register=(3, "q"), index=0>, <Qubit register=(3, "q"), index=1>, <Qubit register=(3, "q"), index=2>), clbits=())
]
"""