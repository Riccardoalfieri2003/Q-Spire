import ast
import re
import sys
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import importlib.util

class QuantumCircuitAnalyzer:
    def __init__(self):
        self.circuit_operations = defaultdict(list)
        self.circuit_names = []
        self.all_circuits = {}  # Store all circuits for subcircuit analysis
        
    def analyze_file(self, filepath: str, debug: bool = False) -> Dict[str, List[Dict]]:
        """
        Analyze a Python file containing quantum circuits and extract operation details.
        
        Args:
            filepath: Path to the Python file to analyze
            debug: If True, print debugging information
            
        Returns:
            Dictionary mapping circuit names to lists of operation details
        """
        # Read the source code
        with open(filepath, 'r') as f:
            source_code = f.read()
        
        if debug:
            print(f"Analyzing file: {filepath}")
        
        # Parse the AST to find circuit-related operations
        tree = ast.parse(source_code)
        
        # Find all QuantumCircuit variables
        circuit_vars = self._find_circuit_variables(tree)
        if debug:
            print(f"Found circuit variables: {circuit_vars}")
        
        # Execute the code to get actual circuit data
        circuits_data = self._execute_and_extract_circuits(filepath, circuit_vars, debug)
        self.all_circuits = circuits_data  # Store for subcircuit analysis
        
        if debug:
            print(f"Extracted circuits: {list(circuits_data.keys())}")
        
        # Map operations to source locations
        results = {}
        for circuit_name, circuit in circuits_data.items():
            if debug:
                print(f"Processing circuit: {circuit_name}")
            operations = self._map_operations_to_source(
                circuit, source_code, circuit_name, debug
            )
            results[circuit_name] = operations
            
        return results
    
    def _find_circuit_variables(self, tree: ast.AST) -> List[str]:
        """Find all QuantumCircuit variable names in the AST."""
        circuit_vars = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Check if it's a QuantumCircuit assignment
                        if isinstance(node.value, ast.Call):
                            if (isinstance(node.value.func, ast.Name) and 
                                node.value.func.id == 'QuantumCircuit') or \
                               (isinstance(node.value.func, ast.Attribute) and 
                                node.value.func.attr == 'QuantumCircuit'):
                                circuit_vars.append(target.id)
        
        return circuit_vars
    
    def _execute_and_extract_circuits(self, filepath: str, circuit_vars: List[str], debug: bool = False) -> Dict[str, Any]:
        """Execute the file and extract quantum circuit objects."""
        # Create a module from the file
        spec = importlib.util.spec_from_file_location("quantum_module", filepath)
        module = importlib.util.module_from_spec(spec)
        
        # Execute the module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Error executing module: {e}")
            return {}
        
        # Extract circuit objects
        circuits = {}
        for var_name in circuit_vars:
            if hasattr(module, var_name):
                circuit = getattr(module, var_name)
                # More robust check for quantum circuit objects
                if (hasattr(circuit, 'data') and 
                    (hasattr(circuit, 'qubits') or hasattr(circuit, 'num_qubits'))):
                    circuits[var_name] = circuit
                    if debug:
                        print(f"Found circuit {var_name} with {len(circuit.data) if hasattr(circuit.data, '__len__') else 'unknown number of'} operations")
        
        # Also check for any other quantum circuit objects in globals
        for name, obj in vars(module).items():
            if (hasattr(obj, 'data') and 
                (hasattr(obj, 'qubits') or hasattr(obj, 'num_qubits')) and 
                name not in circuits and 
                not name.startswith('_')):  # Skip private variables
                circuits[name] = obj
                if debug:
                    print(f"Found additional circuit {name}")
                
        return circuits
    
    def _get_subcircuit_operations_detailed(self, operation_name: str) -> List[Dict]:
        """Extract detailed operations from a subcircuit."""
        # Remove leading/trailing spaces from operation name
        clean_name = operation_name.strip()
        
        # Find the corresponding subcircuit
        for circuit_name, circuit in self.all_circuits.items():
            if hasattr(circuit, 'name') and circuit.name and circuit.name.strip() == clean_name:
                # Extract detailed operations from this subcircuit
                operations = []
                try:
                    circuit_data = list(circuit.data)
                    for instruction in circuit_data:
                        op_info = {
                            'name': instruction.operation.name,
                            'qubits': [q._index for q in instruction.qubits],
                            'clbits': [c._index for c in instruction.clbits] if instruction.clbits else [],
                            'params': list(instruction.operation.params) if hasattr(instruction.operation, 'params') else []
                        }
                        operations.append(op_info)
                    return operations
                except:
                    # Fallback to original name
                    return [{'name': clean_name, 'qubits': [], 'clbits': [], 'params': []}]
        
        # If no matching subcircuit found, return the original operation name
        return [{'name': clean_name, 'qubits': [], 'clbits': [], 'params': []}]
    
    def _map_operations_to_source(self, circuit, source_code: str, circuit_name: str, debug: bool = False) -> List[Dict]:
        """Map circuit operations to their source code locations."""
        lines = source_code.split('\n')
        operations = []
        
        # Get circuit data - handle different Qiskit versions
        try:
            # Try to access data as an iterable
            circuit_data = list(circuit.data)
        except (TypeError, AttributeError):
            try:
                # Try to access data as a property and convert to list
                data_property = circuit.data
                circuit_data = list(data_property) if data_property else []
            except:
                # Fallback: try to iterate through the circuit directly
                circuit_data = []
                try:
                    for instruction in circuit:
                        circuit_data.append(instruction)
                except:
                    return []
        
        if debug:
            print(f"Circuit {circuit_name} has {len(circuit_data)} operations")
        
        # Find lines that contain circuit operations
        operation_lines = []
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if (circuit_name in line and 
                any(method in line for method in ['append', 'h', 'cx', 'measure', 'barrier', 'x', 'y', 'z', 'rx', 'ry', 'rz'])):
                # Skip lines that are part of other circuit definitions
                # Check if this line is defining operations ON the circuit_name, not operations WITHIN another circuit
                if f'{circuit_name}.' in line or f'{circuit_name} =' in line or f'({circuit_name},' in line or f'({circuit_name})' in line:
                    operation_lines.append((i, line))
        
        if debug:
            print(f"Found {len(operation_lines)} operation lines in source")
        
        # Map each operation in circuit.data to source locations
        op_index = 0
        for circuit_instruction in circuit_data:
            operation = circuit_instruction.operation
            qubits = circuit_instruction.qubits
            clbits = circuit_instruction.clbits
            
            # Find the most likely source line for this operation
            source_info = self._find_operation_source_location(
                operation, qubits, clbits, operation_lines, lines, op_index
            )

            # Get the detailed operations from subcircuits
            subcircuit_ops = self._get_subcircuit_operations_detailed(operation.name)
            
            # Check if this is a direct operation (not from a subcircuit)
            is_direct_operation = len(subcircuit_ops) == 1 and subcircuit_ops[0]['name'] == operation.name.strip()
            
            # Create separate entries for each operation in the subcircuit
            for sub_op in subcircuit_ops:
                # Map subcircuit qubit indices to actual circuit qubit indices
                mapped_qubits = []
                if sub_op['qubits']:  # If subcircuit specifies qubits
                    for sub_qubit_idx in sub_op['qubits']:
                        if sub_qubit_idx < len(qubits):
                            mapped_qubits.append(qubits[sub_qubit_idx]._index)
                else:  # If no qubits specified in subcircuit, use all qubits from the main operation (in order)
                    mapped_qubits = [qubit._index for qubit in qubits]
                    
                # Map subcircuit clbit indices to actual circuit clbit indices  
                mapped_clbits = []
                if sub_op['clbits']:  # If subcircuit specifies clbits
                    for sub_clbit_idx in sub_op['clbits']:
                        if sub_clbit_idx < len(clbits):
                            mapped_clbits.append(clbits[sub_clbit_idx]._index)
                else:  # If no clbits specified in subcircuit, use all clbits from the main operation (in order)
                    mapped_clbits = [clbit._index for clbit in clbits] if clbits else []

                # Get parameters - prioritize main operation params for direct operations
                if is_direct_operation and hasattr(operation, 'params') and operation.params:
                    params = list(operation.params)
                else:
                    params = sub_op.get('params', [])

                try: 
                    source_line = eval(source_info["source_line"])
                    operation_info = {
                        'operation_name': sub_op['name'],
                        'qubits_affected': mapped_qubits,
                        'clbits_affected': mapped_clbits,
                        'row': source_line['row'],
                        'column_start': source_line['column_start'],
                        'column_end': source_line['column_end'],
                        'source_line': source_line['source_line']
                    }
                    if params:  # Only add params if they exist
                        operation_info['params'] = params
                    
                except:
                    try: 
                        source_line = eval(source_info["source_line"][:-1])
                        operation_info = {
                            'operation_name': sub_op['name'],
                            'qubits_affected': mapped_qubits,
                            'clbits_affected': mapped_clbits,
                            'row': source_line['row'],
                            'column_start': source_line['column_start'],
                            'column_end': source_line['column_end'],
                            'source_line': source_line['source_line']
                        }
                        if params:  # Only add params if they exist
                            operation_info['params'] = params
            
                    except:
                        operation_info = {
                            'operation_name': sub_op['name'],
                            'qubits_affected': mapped_qubits,
                            'clbits_affected': mapped_clbits,
                            'row': source_info['row'],
                            'column_start': source_info['column_start'],
                            'column_end': source_info['column_end']
                        }
                        if params:  # Only add params if they exist
                            operation_info['params'] = params
                
                operations.append(operation_info)
            
            op_index += 1
        
        return operations
    
    def _find_operation_source_location(self, operation, qubits, clbits, 
                                      operation_lines: List[Tuple[int, str]], 
                                      all_lines: List[str], op_index: int) -> Dict:
        """Find the source location for a specific operation."""
        
        # Default fallback
        default_result = {
            'row': 0,
            'column_start': 0,
            'column_end': 0,
            'source_line': ''
        }
        
        if not operation_lines:
            return default_result
        
        # Try to match operation type and qubits to source lines
        op_name = operation.name.lower()
        qubit_indices = [q._index for q in qubits]
        
        best_match = None
        best_score = -1
        
        for line_num, line in operation_lines:
            score = 0
            line_lower = line.lower()
            
            # Score based on operation name matching
            if op_name == 'h' and 'hadamard' in line_lower:
                score += 10
            elif op_name == 'm' and ('measure' in line_lower):
                score += 10
            elif op_name == 'barrier' and 'barrier' in line_lower:
                score += 10
            elif op_name in line_lower:
                score += 5
            
            # Score based on qubit indices appearing in line
            for qubit_idx in qubit_indices:
                if str(qubit_idx) in line or f'[{qubit_idx}]' in line:
                    score += 3
            
            # Prefer lines that haven't been matched yet (rough heuristic)
            if op_index < len(operation_lines):
                if line_num >= op_index:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = (line_num, line)
        
        if best_match:
            line_num, line = best_match
            
            # Find column positions
            stripped_line = line.lstrip()
            column_start = len(line) - len(stripped_line)
            column_end = len(line.rstrip())
            
            return {
                'row': line_num + 1,  # 1-indexed
                'column_start': column_start,
                'column_end': column_end,
                'source_line': line.strip()
            }
        
        return default_result


def analyze_quantum_file(input_file: str, output_file: str = None, debug: bool = False):
    """
    Analyze a quantum circuit file and optionally save results.
    
    Args:
        input_file: Path to Python file containing quantum circuits
        output_file: Optional path to save analysis results
        debug: If True, print debugging information
    """
    analyzer = QuantumCircuitAnalyzer()
    results = analyzer.analyze_file(input_file, debug)
    
    # Save to file if requested
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

    import pprint
    pprint.pp(results)
    
    return results


# Example of how to use it
if __name__ == "__main__":
    analyze_quantum_file("test/ROC/ROCCode.py", None, debug=False)