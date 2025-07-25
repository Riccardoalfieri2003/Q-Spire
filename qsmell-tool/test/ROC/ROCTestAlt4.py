import ast
import re
import sys
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import importlib.util
import copy

"""
Unica cosa da cambiare: 
- i qubit dell'operazione barrier (in generale quelle che non hanno qubit come argomento, e che colpiscono tutti)
- columns della detection (le rows vanno bene)
"""

"""
Improved QuantumCircuitAnalyzer that tracks dynamic subcircuit construction
"""

class QuantumCircuitAnalyzer:
    def __init__(self):
        self.circuit_operations = defaultdict(list)
        self.circuit_names = []
        self.all_circuits = {}  # Store all circuits for subcircuit analysis
        self.subcircuit_states = {}  # Track subcircuit states during execution
        
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
        
        # Simulate execution step by step to track dynamic subcircuit construction
        results = self._simulate_execution_with_tracking(filepath, source_code, circuit_vars, debug)
            
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
    
    def _simulate_execution_with_tracking(self, filepath: str, source_code: str, circuit_vars: List[str], debug: bool = False) -> Dict[str, List[Dict]]:
        """Simulate execution step by step to track dynamic subcircuit construction."""
        
        # Remove comments and string literals from source code
        cleaned_source_code = self._remove_comments_and_strings(source_code)
        
        # Parse the source code into an AST
        tree = ast.parse(source_code)
        lines = cleaned_source_code.split('\n')
        
        # Initialize tracking structures
        results = {var: [] for var in circuit_vars}
        subcircuit_snapshots = {}  # Store snapshots of subcircuits at different points
        
        # Execute the file first to get the final circuits for reference
        spec = importlib.util.spec_from_file_location("quantum_module", filepath)
        module = importlib.util.module_from_spec(spec)
        
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Error executing module: {e}")
            return results
        
        # Get the final circuits
        final_circuits = {}
        for var_name in circuit_vars:
            if hasattr(module, var_name):
                circuit = getattr(module, var_name)
                if (hasattr(circuit, 'data') and 
                    (hasattr(circuit, 'qubits') or hasattr(circuit, 'num_qubits'))):
                    final_circuits[var_name] = circuit
        
        # Now simulate the execution step by step
        operation_lines = self._extract_operation_sequence(cleaned_source_code, circuit_vars)
        
        if debug:
            print(f"Found {len(operation_lines)} operation lines")
            for i, (line_num, line, op_type, details) in enumerate(operation_lines):
                print(f"  {i}: Line {line_num + 1}: {line.strip()} -> {op_type}")
        
        # Initialize subcircuit states
        for var_name in circuit_vars:
            subcircuit_snapshots[var_name] = []  # List of operations at each point in time
        
        # Process each operation in sequence
        for line_num, line, op_type, details in operation_lines:
            if op_type == 'append':
                # Handle append operations
                main_circuit = details['main_circuit']
                subcircuit_name = details['subcircuit']
                qubits = details.get('qubits', [])
                
                # Get current state of the subcircuit
                current_subcircuit_ops = subcircuit_snapshots.get(subcircuit_name, [])
                
                if debug:
                    print(f"Appending {subcircuit_name} (with {len(current_subcircuit_ops)} ops) to {main_circuit} on qubits {qubits}")
                
                # Add operations from the current state of the subcircuit
                for sub_op in current_subcircuit_ops:
                    # Map subcircuit qubits to main circuit qubits
                    mapped_qubits = self._map_qubits(sub_op['qubits_affected'], qubits)
                    
                    operation_info = {
                        'operation_name': sub_op['operation_name'],
                        'qubits_affected': mapped_qubits,
                        'clbits_affected': sub_op.get('clbits_affected', []),
                        'row': line_num + 1,
                        'column_start': details.get('column_start', 0),
                        'column_end': details.get('column_end', 0),
                        'source_line': lines[line_num].strip() if line_num < len(lines) else ''
                    }
                    
                    if 'params' in sub_op:
                        operation_info['params'] = sub_op['params']
                    
                    results[main_circuit].append(operation_info)
                
            elif op_type == 'direct_operation':
                # Handle direct operations on circuits
                circuit_name = details['circuit']
                operation_name = details['operation']
                qubits = details.get('qubits', [])
                params = details.get('params', [])
                
                operation_info = {
                    'operation_name': operation_name,
                    'qubits_affected': qubits,
                    'clbits_affected': details.get('clbits', []),
                    'row': line_num + 1,
                    'column_start': details.get('column_start', 0),
                    'column_end': details.get('column_end', 0),
                    'source_line': lines[line_num].strip() if line_num < len(lines) else ''
                }
                
                if params:
                    operation_info['params'] = params
                
                # Add to the appropriate circuit
                if circuit_name in results:
                    results[circuit_name].append(operation_info)
                
                # If this is a subcircuit, also track its state
                if circuit_name in subcircuit_snapshots:
                    subcircuit_snapshots[circuit_name].append(operation_info)
        
        return results
    
    def _extract_operation_sequence(self, source_code: str, circuit_vars: List[str]) -> List[Tuple[int, str, str, Dict]]:
        """Extract the sequence of operations from source code in execution order."""
        lines = source_code.split('\n')
        operations = []
        
        # Parse the AST to handle loops and control structures
        tree = ast.parse(source_code)
        
        # For now, we'll use a simplified approach that processes lines sequentially
        # and expands loops based on their range
        operations = self._process_lines_with_loops(lines, circuit_vars)
        
        return operations
    
    def _process_lines_with_loops(self, lines: List[str], circuit_vars: List[str]) -> List[Tuple[int, str, str, Dict]]:
        """Process lines and expand loops to track execution sequence."""
        operations = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Check for for loops
            for_match = re.match(r'for\s+(\w+)\s+in\s+range\s*\(\s*(\d+)\s*\)\s*:', line)
            if for_match:
                loop_var = for_match.group(1)
                loop_count = int(for_match.group(2))
                
                # Find the loop body
                loop_body_start = i + 1
                loop_body_end = self._find_loop_end(lines, i)
                loop_body = lines[loop_body_start:loop_body_end]
                
                # Expand the loop
                for loop_iteration in range(loop_count):
                    for body_line_idx, body_line in enumerate(loop_body):
                        # Replace loop variable with current iteration value
                        expanded_line = body_line.replace(f'[{loop_var}]', f'[{loop_iteration}]')
                        expanded_line = expanded_line.replace(f'({loop_var})', f'({loop_iteration})')
                        
                        actual_line_num = loop_body_start + body_line_idx
                        op_info = self._parse_operation_line(expanded_line, actual_line_num, circuit_vars)
                        if op_info:
                            operations.append(op_info)
                
                i = loop_body_end
                continue
            
            # Process regular lines
            op_info = self._parse_operation_line(line, i, circuit_vars)
            if op_info:
                operations.append(op_info)
            
            i += 1
        
        return operations
    
    def _find_loop_end(self, lines: List[str], loop_start: int) -> int:
        """Find the end of a loop body based on indentation."""
        if loop_start + 1 >= len(lines):
            return len(lines)
        
        loop_indent = len(lines[loop_start]) - len(lines[loop_start].lstrip())
        
        for i in range(loop_start + 1, len(lines)):
            line = lines[i]
            if line.strip() == '':
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= loop_indent:
                return i
        
        return len(lines)
    
    def _parse_operation_line(self, line: str, line_num: int, circuit_vars: List[str]) -> Tuple[int, str, str, Dict]:
        """Parse a single line to extract operation information."""
        line_stripped = line.strip()
        
        for circuit_name in circuit_vars:
            if circuit_name in line_stripped:
                # Check for append operations
                append_match = re.search(rf'{re.escape(circuit_name)}\.append\s*\(\s*(\w+)\s*,\s*\[([^\]]+)\]\s*\)', line_stripped)
                if append_match:
                    subcircuit_name = append_match.group(1)
                    qubits_str = append_match.group(2)
                    qubits = [int(q.strip()) for q in qubits_str.split(',') if q.strip().isdigit()]
                    
                    return (line_num, line, 'append', {
                        'main_circuit': circuit_name,
                        'subcircuit': subcircuit_name,
                        'qubits': qubits,
                        'column_start': append_match.start(),
                        'column_end': append_match.end()
                    })
                
                # Check for direct operations
                op_match = re.search(rf'{re.escape(circuit_name)}\.(\w+)\s*\(([^)]*)\)', line_stripped)
                if op_match:
                    operation_name = op_match.group(1)
                    params_str = op_match.group(2)
                    
                    # Parse parameters and qubits
                    qubits, params = self._parse_operation_params(params_str)
                    
                    return (line_num, line, 'direct_operation', {
                        'circuit': circuit_name,
                        'operation': operation_name,
                        'qubits': qubits,
                        'params': params,
                        'column_start': op_match.start(),
                        'column_end': op_match.end()
                    })
        
        return None
    
    def _parse_operation_params(self, params_str: str) -> Tuple[List[int], List]:
        """Parse operation parameters to extract qubits and other parameters."""
        if not params_str.strip():
            return [], []
        
        qubits = []
        params = []
        
        # Split parameters
        parts = [p.strip() for p in params_str.split(',')]
        
        for part in parts:
            # Check for qubit specifications
            if '[' in part and ']' in part:
                # Extract numbers from brackets
                bracket_content = re.search(r'\[([^\]]+)\]', part)
                if bracket_content:
                    qubit_nums = [int(q.strip()) for q in bracket_content.group(1).split(',') if q.strip().isdigit()]
                    qubits.extend(qubit_nums)
            elif part.isdigit():
                qubits.append(int(part))
            else:
                # This might be a parameter (like phi)
                if part and not part.isspace():
                    params.append(part)
        
        return qubits, params
    
    def _remove_comments_and_strings(self, source_code: str) -> str:
        """Remove multi-line comments (triple-quoted strings) and hash comments from source code."""
        lines = source_code.split('\n')
        cleaned_lines = []
        
        in_triple_quote_single = False
        in_triple_quote_double = False
        
        for line_num, line in enumerate(lines):
            cleaned_line = ""
            j = 0
            
            while j < len(line):
                # Check for triple quotes first (they take precedence)
                if j <= len(line) - 3:
                    three_chars = line[j:j+3]
                    
                    if three_chars == '"""':
                        if in_triple_quote_double:
                            # End of triple double quote block
                            in_triple_quote_double = False
                            j += 3
                            continue
                        elif not in_triple_quote_single:
                            # Start of triple double quote block
                            in_triple_quote_double = True
                            j += 3
                            continue
                    
                    elif three_chars == "'''":
                        if in_triple_quote_single:
                            # End of triple single quote block
                            in_triple_quote_single = False
                            j += 3
                            continue
                        elif not in_triple_quote_double:
                            # Start of triple single quote block
                            in_triple_quote_single = True
                            j += 3
                            continue
                
                # If we're inside a multi-line string (comment), skip everything
                if in_triple_quote_single or in_triple_quote_double:
                    j += 1
                    continue
                
                # Handle single-line comments (only if not in any string)
                if line[j] == '#':
                    # Check if we're inside a regular string first
                    in_string = False
                    quote_char = None
                    for k in range(j):
                        char = line[k]
                        if char in ['"', "'"] and (k == 0 or line[k-1] != '\\'):
                            if not in_string:
                                in_string = True
                                quote_char = char
                            elif char == quote_char:
                                in_string = False
                                quote_char = None
                    
                    # Only treat as comment if not inside a string
                    if not in_string:
                        break  # Rest of line is a comment, skip it
                
                # Regular character - add to cleaned line
                cleaned_line += line[j]
                j += 1
            
            # If the entire line was inside a multi-line comment, make it empty
            if in_triple_quote_single or in_triple_quote_double:
                cleaned_lines.append("")
            else:
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def _map_qubits(self, subcircuit_qubits: List[int], target_qubits: List[int]) -> List[int]:
        """Map subcircuit qubit indices to main circuit qubit indices."""
        if not subcircuit_qubits:
            return target_qubits
        
        mapped = []
        for sub_qubit in subcircuit_qubits:
            if sub_qubit < len(target_qubits):
                mapped.append(target_qubits[sub_qubit])
        
        return mapped if mapped else target_qubits


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