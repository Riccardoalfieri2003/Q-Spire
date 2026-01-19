from smells.utils.OperationCircuitTracker import analyze_quantum_file
from collections import defaultdict
from smells.Detector import Detector
from smells.IQ.IQ import IQ
from smells.utils.config_loader import get_detector_option




def create_circuit_batches(operations):
    """
    Convert a sequential list of quantum operations into parallel execution batches.
    
    Each batch contains operations that can be executed in parallel (don't conflict on qubits).
    The batches represent the actual execution order in the quantum circuit.
    
    Args:
        operations: List of operation dictionaries with 'qubits_affected' field
        
    Returns:
        dict: Dictionary where keys are batch numbers and values are lists of operations
    """
    
    if not operations:
        return {}
    
    batches = {}
    batch_number = 1
    
    # Keep track of the last batch where each qubit was used
    qubit_last_batch = {}
    
    # Process each operation in order
    for operation in operations:
        operation_qubits = operation['qubits_affected']
        
        # Find the minimum batch this operation can be placed in
        # It must be after the last batch that used any of its qubits
        min_batch = 1
        for qubit in operation_qubits:
            if qubit in qubit_last_batch:
                min_batch = max(min_batch, qubit_last_batch[qubit] + 1)
        
        # Try to place the operation in the earliest possible batch
        # where it doesn't conflict with other operations in that batch
        placed = False
        current_batch_num = min_batch
        
        while not placed:
            # Check if this batch exists and if there's a conflict
            if current_batch_num not in batches:
                batches[current_batch_num] = []
            
            # Check for conflicts with existing operations in this batch
            conflict = False
            for existing_op in batches[current_batch_num]:
                existing_qubits = set(existing_op['qubits_affected'])
                if set(operation_qubits).intersection(existing_qubits):
                    conflict = True
                    break
            
            if not conflict:
                # Place the operation in this batch
                batches[current_batch_num].append(operation)
                
                # Update the last batch for all affected qubits
                for qubit in operation_qubits:
                    qubit_last_batch[qubit] = current_batch_num
                
                placed = True
            else:
                # Try the next batch
                current_batch_num += 1
    
    # Clean up empty batches and renumber
    final_batches = {}
    batch_counter = 1
    for batch_num in sorted(batches.keys()):
        if batches[batch_num]:  # Only include non-empty batches
            final_batches[batch_counter] = batches[batch_num]
            batch_counter += 1
    
    return final_batches

def print_circuit_batches(batches):
    """Helper function to print the batches in a readable format"""
    for batch_num, operations in batches.items():
        print(f"\nBatch {batch_num}:")
        for op in operations:
            qubits_str = ', '.join(map(str, op['qubits_affected']))
            print(f"  - {op['operation_name']} on qubit(s) [{qubits_str}] (row {op['row']})")



def detect_iq_smell_from_batches(batches, max_distance, circuit_name):
    
    """
    Detect IQ (Idle Qubit) smell using circuit batches.
    
    For each qubit, check if there are too many batches between its FIRST and SECOND operations only.
    
    Args:
        batches: Dictionary with batch numbers as keys and lists of operations as values
        max_distance: Maximum allowed batch distance between operations on same qubit
        circuit_name: Name of the circuit being analyzed
        
    Returns:
        list: List of IQ smell objects
    """
    smells = []
    
    # Track the first two operations for each qubit
    qubit_operations = {}
    
    # Process each batch in order to find first and second operations per qubit
    for batch_num in sorted(batches.keys()):
        operations = batches[batch_num]
        
        for operation in operations:
            for qubit in operation['qubits_affected']:
                # Initialize tracking for this qubit if not exists
                if qubit not in qubit_operations:
                    qubit_operations[qubit] = []
                
                # Only store first two operations
                if len(qubit_operations[qubit]) < 2:
                    qubit_operations[qubit].append({
                        'batch': batch_num,
                        'operation': operation
                    })
    
    # Check for IQ smell on each qubit (only between first and second operation)
    for qubit, operations in qubit_operations.items():
        if len(operations) < 2:
            # Need at least 2 operations to check distance
            continue
            
        # Only check distance between first and second operation
        first_batch = operations[0]['batch']
        second_batch = operations[1]['batch']
        
        # Calculate batch distance (number of batches between operations)
        distance = second_batch - first_batch - 1
        
        if distance > max_distance:
            # Get operation details for smell creation (using second operation)
            second_op = operations[1]['operation']
            
            smell = IQ(
                row=second_op['row'],
                column_start=second_op['column_start'],
                column_end=second_op['column_end']+1,
                circuit_name=circuit_name,
                qubit=qubit,
                operation_distance=distance,
                operation_name=second_op['operation_name']
            )
            smells.append(smell)
    
    return smells





@Detector.register(IQ)
class IQDetector(Detector):

    def __init__(self, smell_cls):
        super().__init__(smell_cls)
        self.circuit_tracker = None
        self.operation_tracker = None
        self.position_tracker = None


    def detect(self, file):
        smells = []

        circuits = analyze_quantum_file(file)

        max_distance = get_detector_option("IQ", "max_distance", fallback=2)

        for circuit in circuits: 

            circuit_batches = create_circuit_batches(circuits[circuit])

            iq_smells = detect_iq_smell_from_batches(circuit_batches, max_distance=max_distance, circuit_name=circuit)
            for smell in iq_smells:
                smells.append(smell)
        

        return smells
