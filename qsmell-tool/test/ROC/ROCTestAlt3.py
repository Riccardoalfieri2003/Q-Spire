import ast
import re
import sys
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import importlib.util

"""
Usare la versione 4, migliore
"""

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
            operations = self._map_operations_to_source_improved(
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
    
    def _find_operation_lines_in_source(self, source_code: str, circuit_name: str) -> List[Tuple[int, str, Dict]]:
        """Find all lines that contain quantum operations and parse their details."""
        lines = source_code.split('\n')
        operation_lines = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith('#'):
                continue
                
            # Check if this line contains circuit operations
            if circuit_name in line:
                # Look for method calls on the circuit
                operation_match = re.search(rf'{re.escape(circuit_name)}\.(\w+)\s*\([^)]*\)', line)
                if operation_match:
                    method_name = operation_match.group(1)
                    
                    # Extract qubit indices from the line
                    qubit_indices = self._extract_qubit_indices_from_line(line)
                    
                    # Find column positions for the operation
                    method_start = operation_match.start()
                    method_end = operation_match.end()
                    
                    operation_info = {
                        'method': method_name,
                        'qubits': qubit_indices,
                        'column_start': method_start,
                        'column_end': method_end,
                        'full_match': operation_match.group(0)
                    }
                    
                    operation_lines.append((i, line, operation_info))
        
        return operation_lines
    
    def _extract_qubit_indices_from_line(self, line: str) -> List[int]:
        """Extract qubit indices from a line of code."""
        qubit_indices = []
        
        # Look for patterns like [0], [1, 2], or (0, 1)
        bracket_matches = re.findall(r'\[([^\]]+)\]', line)
        paren_matches = re.findall(r'\(([^)]+)\)', line)
        
        for match in bracket_matches + paren_matches:
            # Split by comma and try to parse as integers
            parts = [part.strip() for part in match.split(',')]
            for part in parts:
                try:
                    # Handle simple integer
                    if part.isdigit():
                        qubit_indices.append(int(part))
                    # Handle variable references (like 'j' in loops)
                    elif part.isalpha() and len(part) == 1:
                        # This is likely a loop variable, we'll handle it differently
                        pass
                except ValueError:
                    continue
        
        return qubit_indices
    
    def _map_operations_to_source_improved(self, circuit, source_code: str, circuit_name: str, debug: bool = False) -> List[Dict]:
        """Improved mapping of circuit operations to their source code locations."""
        lines = source_code.split('\n')
        operations = []
        
        # Get circuit data
        try:
            circuit_data = list(circuit.data)
        except (TypeError, AttributeError):
            try:
                data_property = circuit.data
                circuit_data = list(data_property) if data_property else []
            except:
                circuit_data = []
                try:
                    for instruction in circuit:
                        circuit_data.append(instruction)
                except:
                    return []
        
        if debug:
            print(f"Circuit {circuit_name} has {len(circuit_data)} operations")
        
        # Find all operation lines in source
        operation_lines = self._find_operation_lines_in_source(source_code, circuit_name)
        
        if debug:
            print(f"Found {len(operation_lines)} operation lines in source")
            for line_num, line, info in operation_lines:
                print(f"  Line {line_num + 1}: {line.strip()} -> {info}")
        
        # Create a mapping strategy
        operation_index = 0
        used_lines = set()  # Track which lines we've already used for NON-LOOP operations only
        
        for circuit_instruction in circuit_data:
            operation = circuit_instruction.operation
            qubits = circuit_instruction.qubits
            clbits = circuit_instruction.clbits
            
            # Find the best matching source line for this operation
            best_match = self._find_best_source_match(
                operation, qubits, clbits, operation_lines, operation_index, used_lines, debug
            )
            
            if best_match:
                line_num, line, op_info = best_match
                # Only mark non-loop lines as used
                if not self._line_can_generate_multiple_operations(line):
                    used_lines.add((line_num, operation.name))
            else:
                # Fallback: create a default entry
                line_num = 0
                line = ""
                op_info = {
                    'method': operation.name,
                    'qubits': [],
                    'column_start': 0,
                    'column_end': 0,
                    'full_match': operation.name
                }
            
            # Get detailed operations from subcircuits
            subcircuit_ops = self._get_subcircuit_operations_detailed(operation.name)
            
            # Check if this is a direct operation
            is_direct_operation = len(subcircuit_ops) == 1 and subcircuit_ops[0]['name'] == operation.name.strip()
            
            # Create entries for each operation in the subcircuit
            for sub_op in subcircuit_ops:
                # Map subcircuit qubit indices to actual circuit qubit indices
                mapped_qubits = []
                if sub_op['qubits']:
                    for sub_qubit_idx in sub_op['qubits']:
                        if sub_qubit_idx < len(qubits):
                            mapped_qubits.append(qubits[sub_qubit_idx]._index)
                else:
                    mapped_qubits = [qubit._index for qubit in qubits]
                    
                # Map subcircuit clbit indices
                mapped_clbits = []
                if sub_op['clbits']:
                    for sub_clbit_idx in sub_op['clbits']:
                        if sub_clbit_idx < len(clbits):
                            mapped_clbits.append(clbits[sub_clbit_idx]._index)
                else:
                    mapped_clbits = [clbit._index for clbit in clbits] if clbits else []

                # Get parameters
                if is_direct_operation and hasattr(operation, 'params') and operation.params:
                    params = list(operation.params)
                else:
                    params = sub_op.get('params', [])

                operation_info = {
                    'operation_name': sub_op['name'],
                    'qubits_affected': mapped_qubits,
                    'clbits_affected': mapped_clbits,
                    'row': line_num + 1,  # 1-indexed
                    'column_start': op_info['column_start'],
                    'column_end': op_info['column_end'],
                    'source_line': line.strip() if line else ''
                }
                
                if params:
                    operation_info['params'] = params
                
                operations.append(operation_info)
            
            operation_index += 1
        
        return operations
    
    def _find_best_source_match(self, operation, qubits, clbits, operation_lines, 
                               operation_index, used_lines, debug=False) -> Tuple[int, str, Dict]:
        """Find the best matching source line for an operation."""
        
        op_name = operation.name.lower()
        qubit_indices = [q._index for q in qubits]
        
        if debug:
            print(f"Looking for operation: {op_name} on qubits {qubit_indices}")
        
        # Find lines that match the operation type
        candidates = []
        for line_num, line, op_info in operation_lines:
            method_name = op_info['method'].lower()
            
            # Check if operation names match
            if self._operations_match(op_name, method_name):
                # For operations in loops, the same line can generate multiple operations
                if self._line_can_generate_multiple_operations(line):
                    # This line can generate multiple operations (like in a loop)
                    # Always allow matching to loop lines regardless of used_lines
                    candidates.append((line_num, line, op_info, 100))  # High priority
                else:
                    # For non-loop operations, check if already used
                    if (line_num, operation.name) not in used_lines:
                        # Check if qubits match for non-loop operations
                        if self._qubits_match(qubit_indices, op_info['qubits'], line):
                            candidates.append((line_num, line, op_info, 50))
                        else:
                            candidates.append((line_num, line, op_info, 10))  # Lower priority
        
        if candidates:
            # Sort by priority (higher first)
            candidates.sort(key=lambda x: x[3], reverse=True)
            best = candidates[0]
            return (best[0], best[1], best[2])
        
        # If no good match found, return the first available operation line
        for line_num, line, op_info in operation_lines:
            if (line_num, operation.name) not in used_lines:
                return (line_num, line, op_info)
        
        return None
    
    def _operations_match(self, circuit_op_name: str, source_method_name: str) -> bool:
        """Check if circuit operation name matches source method name."""
        # Handle common aliases
        aliases = {
            'h': ['h', 'hadamard'],
            'cx': ['cx', 'cnot'],
            'measure': ['measure', 'm'],
            'x': ['x', 'pauli_x'],
            'y': ['y', 'pauli_y'],
            'z': ['z', 'pauli_z'],
            'barrier': ['barrier']
        }
        
        circuit_op_lower = circuit_op_name.lower()
        source_method_lower = source_method_name.lower()
        
        # Direct match
        if circuit_op_lower == source_method_lower:
            return True
        
        # Check aliases
        for op, alias_list in aliases.items():
            if circuit_op_lower == op and source_method_lower in alias_list:
                return True
            if source_method_lower == op and circuit_op_lower in alias_list:
                return True
        
        return False
    
    def _line_can_generate_multiple_operations(self, line: str) -> bool:
        """Check if a line can generate multiple operations (e.g., in a loop)."""
        # Look for loop indicators
        return 'for ' in line or 'while ' in line or any(var in line for var in ['range(', '[j]', '[i]', '(j)', '(i)'])
    
    def _qubits_match(self, circuit_qubits: List[int], source_qubits: List[int], line: str) -> bool:
        """Check if qubits from circuit match those in source line."""
        if not source_qubits:  # Source line might use variables
            return True  # We can't verify, so assume it matches
        
        return set(circuit_qubits) == set(source_qubits)


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