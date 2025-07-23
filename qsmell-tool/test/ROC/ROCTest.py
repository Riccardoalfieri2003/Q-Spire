"""import ast
from smells.Detector import Detector
from smells.ROC.ROCDetector import ROCDetector
from smells.ROC.ROC import ROC


import ast

class GeneralQiskitOpVisitor(ast.NodeVisitor):
    def __init__(self, circuit_variable_names=None):
        super().__init__()
        self.operations = []
        self.circuit_variable_names = circuit_variable_names or set()

    def visit_Assign(self, node):
        # Track variables that store QuantumCircuit objects
        if isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "QuantumCircuit":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.circuit_variable_names.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check if it's a method call on a circuit variable
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            circuit_var = node.func.value.id
            if circuit_var in self.circuit_variable_names:
                self.operations.append({
                    "circuit": circuit_var,
                    "method": node.func.attr,  # e.g., h, cx, custom_gate
                    "lineno": node.lineno,
                    "col_offset": node.col_offset,
                    "end_lineno": getattr(node, "end_lineno", node.lineno),
                    "end_col_offset": getattr(node, "end_col_offset", node.col_offset + 1),
                    "raw": ast.unparse(node)  # full source of the op
                })
        self.generic_visit(node)





def test_custom_gate_detector():
    
    # Example quantum code with custom gates
    with open("test/ROC/ROCCode.py", "r") as file:
        test_code = file.read()    

    tree = ast.parse(test_code)
    visitor = GeneralQiskitOpVisitor()
    visitor.visit(tree)

    for op in visitor.operations:
        print(op)


if __name__ == "__main__":
    test_custom_gate_detector()"""



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
                    #print(f"Warning: Could not extract data from circuit {circuit_name}")
                    return []
        
        if debug:
            print(f"Circuit {circuit_name} has {len(circuit_data)} operations")
        
        # Find lines that contain circuit operations
        operation_lines = []
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if (circuit_name in line and 
                any(method in line for method in ['append', 'h', 'cx', 'measure', 'barrier', 'x', 'y', 'z', 'rx', 'ry', 'rz'])):
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
            
            operation_info = {
                'operation_name': operation.name,
                'qubits_affected': [qubit._index for qubit in qubits],
                'clbits_affected': [clbit._index for clbit in clbits] if clbits else [],
                #'row': source_info['row'],
                #'column_start': source_info['column_start'],
                #'column_end': source_info['column_end'],
                'source_line': source_info['source_line']
            }
            
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


# Example usage
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
    
    """
    # Print results
    for circuit_name, operations in results.items():
        print(f"\n=== Circuit: {circuit_name} ===")
        for i, op in enumerate(operations):
            print(f"Operation {i+1}:")
            print(f"  Name: {op['operation_name']}")
            print(f"  Qubits: {op['qubits_affected']}")
            if op['clbits_affected']:
                print(f"  Classical bits: {op['clbits_affected']}")
            print(f"  Location: Row {op['row']}, Columns {op['column_start']}-{op['column_end']}")
            print(f"  Source: {op['source_line']}")
            print()
    """
    
    # Save to file if requested
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

    import pprint
    pprint.pp(results)
    #print(results)
    
    return results



# Example of how to use it
if __name__ == "__main__":
    analyze_quantum_file("test/ROC/ROCCode.py", None, debug=False)
    pass