import builtins
import inspect
import math
import textwrap
import tempfile
import runpy
import os

from qiskit import QuantumCircuit
from smells.Detector import Detector

from smells.LC.LC import LC
from smells.utils.OperationCircuitTracker import analyze_quantum_file
from smells.utils.BackendAnalyzer import analyze_circuits_backends_runs
from smells.utils.config_loader import get_detector_option

from smells.utils.CircuitTaker import analyze_quantum_file_circuits

import math
from typing import Dict, List, Tuple, Any

def max_parallelism(circuit):
    """Calculate the maximum number of parallel operations in the circuit."""
    moments = []
    for instr, qargs, cargs in circuit.data:
        qindices = [q._index for q in qargs]
        placed = False
        for moment in moments:
            if not any(q in moment for q in qindices):
                moment.update(qindices)
                placed = True
                break
        if not placed:
            moments.append(set(qindices))
    return max(len(m) for m in moments) if moments else 0

def max_operations_per_qubit(circuit: QuantumCircuit) -> int:
    """Calculate the maximum number of operations on any single qubit in the circuit."""
    qubit_counts = [0] * circuit.num_qubits

    for instruction, qargs, _ in circuit.data:
        for qubit in qargs:
            qubit_index = qubit._index
            qubit_counts[qubit_index] += 1

    return max(qubit_counts)

def get_max_gate_errors(backend_properties) -> dict:
    """
    Returns a dictionary with all gates and their maximum error rates.
    """
    if backend_properties is None:
        return {}
    
    gate_errors = {}
    
    # Iterate through all gates in the backend properties
    for gate in backend_properties.gates:
        gate_name = gate.gate
        
        # Find the gate_error parameter for this gate
        for param in gate.parameters:
            if param.name == 'gate_error':
                error_value = param.value
                
                # Keep track of the maximum error for each gate type
                if gate_name not in gate_errors:
                    gate_errors[gate_name] = error_value
                else:
                    gate_errors[gate_name] = max(gate_errors[gate_name], error_value)
                break
    
    return gate_errors

def get_max_gate_error(backend_properties) -> dict:
    """
    Returns a dictionary with the single gate having the maximum error rate.
    Format: {'gate_name': max_error}
    """
    # Get all gate errors
    all_errors = get_max_gate_errors(backend_properties)
    
    if not all_errors:
        return {}
    
    # Find the gate with maximum error
    max_gate = max(all_errors.items(), key=lambda x: x[1])
    
    return {max_gate[0]: max_gate[1]}

def map_circuits_to_backends(circuits: Dict[str, Any], backends: Dict[str, Dict], runs: List[Dict]) -> List[Tuple]:
    """
    Map circuits to backends based on run executions, similar to your original run_log format.
    
    Returns:
        List of tuples (circuit, backend_instance, circuit_name)
    """
    mapped_runs = []
    
    for run in runs:
        backend_var = run['backend_variable']
        circuits_used = run['circuits_used']
        
        # Get the backend instance
        if backend_var in backends and 'instance' in backends[backend_var]:
            backend_instance = backends[backend_var]['instance']
            
            # Map each circuit used in this run
            for circuit_name in circuits_used:
                if circuit_name in circuits:
                    circuit = circuits[circuit_name]
                    mapped_runs.append((circuit, backend_instance, circuit_name))
    
    return mapped_runs

def detect_with_new_analyzer(self, file):
    """
    Updated detect function using the new analyzer.
    """
    smells = []
    circuits, backends, runs = analyze_circuits_backends_runs(file, debug=False)
    
    # Map circuits to backends based on runs (similar to your original run_log)
    circuit_backend_mappings = map_circuits_to_backends(circuits, backends, runs)
    
    for circuit, backend, circuit_name in circuit_backend_mappings:

        print(circuit)
        try:
            # Get backend properties
            backend_properties = backend.properties()
            if backend_properties is None:
                continue
                
            # Get max gate error
            max_gate_error = get_max_gate_error(backend_properties)
            if not max_gate_error:
                continue
                
            (gate_name, error_value), = max_gate_error.items()
            
            # Calculate your metrics
            l = max_operations_per_qubit(circuit)
            c = max_parallelism(circuit)
            
            likelihood = math.pow(1 - error_value, l * c)
            
            # Heuristic thresholds — adjust as needed
            if likelihood < 0.5:
                backend_class_name = backend.__class__.__name__
                
                smell = LC(
                    likelihood=likelihood,
                    error=max_gate_error,
                    l=l,
                    c=c,
                    backend=backend_class_name,
                    circuit_name=circuit_name,
                    explanation="",
                    suggestion="",
                    circuit_operations=circuit
                )
                
                smells.append(smell)
                
        except Exception as e:
            print(f"Error processing circuit {circuit_name} with backend: {e}")
            continue
    
    return smells

# Alternative approach: Direct iteration over runs
def detect_alternative_approach(self, file):
    """
    Alternative approach: iterate directly over runs and get corresponding circuits/backends.
    """
    smells = []
    circuits, backends, runs = analyze_circuits_backends_runs(file, debug=False)
    
    for run in runs:
        backend_var = run['backend_variable']
        circuits_used = run['circuits_used']
        
        # Get the backend instance and info
        if backend_var not in backends or 'instance' not in backends[backend_var]:
            continue
            
        backend_instance = backends[backend_var]['instance']
        
        try:
            # Get backend properties
            backend_properties = backend_instance.properties()
            if backend_properties is None:
                continue
                
            # Get max gate error
            max_gate_error = get_max_gate_error(backend_properties)
            if not max_gate_error:
                continue
                
            (gate_name, error_value), = max_gate_error.items()
            
            # Process each circuit used in this run
            for circuit_name in circuits_used:
                if circuit_name not in circuits:
                    continue
                    
                circuit = circuits[circuit_name]
                
                # Calculate your metrics
                l = max_operations_per_qubit(circuit)
                c = max_parallelism(circuit)
                
                likelihood = math.pow(1 - error_value, l * c)
                
                # Heuristic thresholds — adjust as needed
                if likelihood < 0.5:
                    backend_class_name = backend_instance.__class__.__name__
                    
                    smell = LC(
                        likelihood=likelihood,
                        error=max_gate_error,
                        l=l,
                        c=c,
                        backend=backend_class_name,
                        circuit_name=circuit_name,
                        explanation="",
                        suggestion=""
                    )
                    
                    smells.append(smell)
                    
        except Exception as e:
            print(f"Error processing run with backend {backend_var}: {e}")
            continue
    
    return smells

# Example of how to access backend information
def explore_backend_structure(backends: Dict[str, Dict]):
    """
    Helper function to explore the structure of backends dictionary.
    """
    for backend_name, backend_info in backends.items():
        print(f"\nBackend: {backend_name}")
        print(f"  Keys: {list(backend_info.keys())}")
        
        if 'instance' in backend_info:
            backend_instance = backend_info['instance']
            print(f"  Instance type: {type(backend_instance).__name__}")
            print(f"  Has properties method: {hasattr(backend_instance, 'properties')}")
            
            try:
                props = backend_instance.properties()
                print(f"  Properties available: {props is not None}")
                if props:
                    print(f"  Number of gates: {len(props.gates) if hasattr(props, 'gates') else 'N/A'}")
            except Exception as e:
                print(f"  Error getting properties: {e}")







@Detector.register(LC)
class LCDetector(Detector):

    
    def detect(self, file):

        smells = []        

        error_threshold = get_detector_option("LC", "gate_error", fallback=0)
        threshold = get_detector_option("LC", "threshold", fallback=0.5)

        if error_threshold==0:

            print("Fallback error gate value")

            circuits, backends, runs = analyze_circuits_backends_runs(file, debug=False)
            mappings = map_circuits_to_backends(circuits, backends, runs)

            for circuit, backend, circuit_name in mappings:
                
                # Test your functions
                try:
                    props = backend.properties()
                    if props:

                        gate_name, max_error = list(get_max_gate_error(props).items())[0]
                        l = max_operations_per_qubit(circuit)
                        c = max_parallelism(circuit)

                        likelihood=math.pow(1-max_error,l*c)

                        # Heuristic thresholds — adjust as needed
                        if likelihood<threshold:
                            
                            backend_class_name = backend.__class__.__name__ 

                            smell = LC(
                                likelihood=likelihood,
                                error={gate_name:max_error},
                                l=l,
                                c=c,
                                backend=backend_class_name,
                                circuit_name=circuit_name,  # Use the captured name
                                explanation="",
                                suggestion=""
                            )

                            smells.append(smell)


                except Exception as e:
                    print(f"  Error: {e}")
            

        else:

            
            
            
            """
            circuits, backends, runs = analyze_circuits_backends_runs(file, debug=False)

            # circuits = analyze_quantum_file_circuits(file, debug=True)
            circuits = analyze_quantum_file(file)

            print(f"\n\n    error gate: {error_threshold}\n    threshold: {threshold}\n      Circuiti: {circuits}\n      file:{file}\n")

            for circuit_name, circuit in circuits.items():

                create_circuit_batches(circuits[circuit])

                gate_name = "CustomGate"
                max_error = error_threshold  # Retrieved from config

                lenght_op = max_operations_per_qubit(circuit)
                parallel_op = max_parallelism(circuit)

                likelihood=math.pow(1-max_error,lenght_op*parallel_op)

                print(f"Likelihood: {likelihood}")

                # Heuristic thresholds — adjust as needed
                if likelihood<threshold:

                    smell = LC(
                        likelihood=likelihood,
                        error={gate_name:max_error},
                        lenght_op=lenght_op,
                        parallel_op=parallel_op,
                        backend="Custom Backend",
                        circuit_name=circuit_name,  # Use the captured name
                        explanation="",
                        suggestion="",
                        circuit=circuit
                    )

                    smells.append(smell)
            """
            
            circuits, backends, runs = analyze_circuits_backends_runs(file, debug=False)

            # circuits = analyze_quantum_file_circuits(file, debug=True)
            circuits = analyze_quantum_file(file)

            #print(f"\n\n    error gate: {error_threshold}\n    threshold: {threshold}\n      Circuiti: {circuits}\n      file:{file}\n")

            for circuit in circuits:

                batches=create_circuit_batches(circuits[circuit])


                gate_name = "CustomGate"
                max_error = error_threshold  # Retrieved from config

                lenght_op, parallel_op = analyze_batches(batches)

                #if lenght_op==0 or parallel_op==0:
                import pprint
                pprint.pp(batches)

                #lenght_op = 
                #parallel_op = 

                likelihood=math.pow(1-max_error,lenght_op*parallel_op)

                print(f"lenght_op, parallel_op: {lenght_op, parallel_op}")
                print(f"Likelihood: {likelihood}")

                # Heuristic thresholds — adjust as needed
                if likelihood<threshold:

                    smell = LC(
                        likelihood=likelihood,
                        error={gate_name:max_error},
                        lenght_op=lenght_op,
                        parallel_op=parallel_op,
                        backend="Custom Backend",
                        circuit_name=circuit,  # Use the captured name
                        explanation="",
                        suggestion="",
                        circuit=circuit
                    )

                    smells.append(smell)






        
        return smells




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


def analyze_batches(batches: Dict[int, List[Dict]], debug: bool = False) -> tuple[int, int]:
    """
    Analyze quantum circuit batches to find:
    1. Max operations on any single qubit
    2. Max operations executed in parallel (largest batch)
    
    Args:
        batches: Dictionary with batch IDs as keys and lists of operation dicts as values
        debug: If True, print debug information about operations
    
    Returns:
        Tuple of (max_ops_per_qubit, max_ops_in_parallel)
    """
    
    # Count operations per qubit (including classical bits)
    qubit_operation_count = {}
    
    for batch_id, operations in batches.items():
        for operation in operations:
            # Get all qubits affected by this operation
            qubits = operation.get('qubits_affected', [])
            clbits = operation.get('clbits_affected', [])
            
            if debug and not qubits and not clbits:
                print(f"Debug: Batch {batch_id}, Operation '{operation.get('operation_name')}' has no qubits or clbits affected")
            
            # Count qubits
            for qubit in qubits:
                qubit_operation_count[qubit] = qubit_operation_count.get(qubit, 0) + 1
            
            # Also count classical bits if no qubits
            for clbit in clbits:
                qubit_operation_count[f"cbit_{clbit}"] = qubit_operation_count.get(f"cbit_{clbit}", 0) + 1
    
    # Get max operations on any single qubit
    max_ops_per_qubit = max(qubit_operation_count.values()) if qubit_operation_count else 0
    
    # Get max operations executed in parallel (largest batch)
    max_ops_in_parallel = max(len(operations) for operations in batches.values()) if batches else 0
    
    if debug:
        print(f"Qubit operation counts: {qubit_operation_count}")
        print(f"Max ops per qubit: {max_ops_per_qubit}, Max ops in parallel: {max_ops_in_parallel}")
    
    return max_ops_per_qubit, max_ops_in_parallel
    """
    Analyze quantum circuit batches to find:
    1. Max operations on any single qubit
    2. Max operations executed in parallel (largest batch)
    
    Args:
        batches: Dictionary with batch IDs as keys and lists of operation dicts as values
    
    Returns:
        Tuple of (max_ops_per_qubit, max_ops_in_parallel)
    """
    
    # Count operations per qubit
    qubit_operation_count = {}
    
    for batch_id, operations in batches.items():
        for operation in operations:
            # Get all qubits affected by this operation
            for qubit in operation['qubits_affected']:
                qubit_operation_count[qubit] = qubit_operation_count.get(qubit, 0) + 1
    
    # Get max operations on any single qubit
    max_ops_per_qubit = max(qubit_operation_count.values()) if qubit_operation_count else 0
    
    # Get max operations executed in parallel (largest batch)
    max_ops_in_parallel = max(len(operations) for operations in batches.values()) if batches else 0
    
    return max_ops_per_qubit, max_ops_in_parallel