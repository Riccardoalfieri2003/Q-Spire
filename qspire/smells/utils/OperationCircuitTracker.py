import ast
import re
import sys
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import importlib.util
import copy

"""
Fixed QuantumCircuitAnalyzer that properly handles measurements in nested loops
"""

class QuantumCircuitAnalyzer:
    def __init__(self):
        self.circuit_operations = defaultdict(list)
        self.circuit_names = []
        self.all_circuits = {}  # Store all circuits for subcircuit analysis
        self.subcircuit_states = {}  # Track subcircuit states during execution
        self.circuit_sizes = {}  # Track the number of qubits in each circuit
        self.register_info = {}  # Track quantum and classical register information
        
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
        with open(filepath, 'r', encoding="utf-8") as f:
            source_code = f.read()
        
        """if debug:
            print(f"Analyzing file: {filepath}")"""
        #debug=True
        # Parse the AST to find circuit-related operations
        tree = ast.parse(source_code)
        
        # Find all QuantumCircuit variables and their sizes, plus register info
        circuit_vars = self._find_circuit_variables(tree, source_code)
        #print(f"circuit_vars: {circuit_vars}")
        self._find_register_variables(tree)
        if debug:
            print(f"Found circuit variables: {circuit_vars}")
            print(f"Circuit sizes: {self.circuit_sizes}")
            print(f"Register info: {self.register_info}")
        
        debug=False
        # Simulate execution step by step to track dynamic subcircuit construction
        results = self._simulate_execution_with_tracking(filepath, source_code, circuit_vars, debug)
            
        return results
    
    def _find_circuit_variables(self, tree: ast.AST, source_code: str) -> List[str]:
        """Find all QuantumCircuit variable names in the AST and extract their sizes."""
        circuit_vars = []
        lines = source_code.split('\n')
        
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
                                
                                # Try to extract the number of qubits
                                if node.value.args:
                                    if isinstance(node.value.args[0], ast.Constant):
                                        self.circuit_sizes[target.id] = node.value.args[0].value
                                    elif isinstance(node.value.args[0], ast.Num):  # For older Python versions
                                        self.circuit_sizes[target.id] = node.value.args[0].n
                                    else:
                                        # Try to extract from source line
                                        line_num = node.lineno - 1
                                        if line_num < len(lines):
                                            line = lines[line_num]
                                            match = re.search(r'QuantumCircuit\s*\(\s*(\d+)', line)
                                            if match:
                                                self.circuit_sizes[target.id] = int(match.group(1))
        
        return circuit_vars
    
    def _find_register_variables(self, tree: ast.AST) -> None:
        """Find all QuantumRegister and ClassicalRegister variables and their sizes."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Check if it's a QuantumRegister or ClassicalRegister assignment
                        if isinstance(node.value, ast.Call):
                            register_type = None
                            if isinstance(node.value.func, ast.Name):
                                if node.value.func.id == 'QuantumRegister':
                                    register_type = 'quantum'
                                elif node.value.func.id == 'ClassicalRegister':
                                    register_type = 'classical'
                            elif isinstance(node.value.func, ast.Attribute):
                                if node.value.func.attr == 'QuantumRegister':
                                    register_type = 'quantum'
                                elif node.value.func.attr == 'ClassicalRegister':
                                    register_type = 'classical'
                            
                            if register_type and node.value.args:
                                # Try to extract the size
                                size = None
                                if isinstance(node.value.args[0], ast.Constant):
                                    size = node.value.args[0].value
                                elif isinstance(node.value.args[0], ast.Num):  # For older Python versions
                                    size = node.value.args[0].n
                                
                                if size is not None:
                                    self.register_info[target.id] = {
                                        'type': register_type,
                                        'size': size
                                    }


    
    def _simulate_execution_with_tracking(self, filepath: str, source_code: str, circuit_vars: List[str], debug: bool = False) -> Dict[str, List[Dict]]:
        """Simulate execution step by step to track dynamic subcircuit construction."""

        def _has_main_block( self, source_code: str) -> bool:
            # Remove comments and strings to avoid false positives
            cleaned_code = self._remove_comments_and_strings(source_code)
            
            # Check for common variations
            patterns = [
                'if __name__ == "__main__"',
                "if __name__ == '__main__'",
                'if __name__=="__main__"',
                "if __name__=='__main__'",
            ]
            
            return any(pattern in cleaned_code for pattern in patterns)
        
        # Remove comments and string literals from source code
        cleaned_source_code = self._remove_comments_and_strings(source_code)
        
        # Parse the source code into an AST
        tree = ast.parse(source_code)
        lines = source_code.split('\n')  # Use original source code to preserve spacing
        
        # Initialize tracking structures
        results = {var: [] for var in circuit_vars}
        subcircuit_snapshots = {}  # Store snapshots of subcircuits at different points
        

        # Read and execute the file with __name__ set to "__main__"
        with open(filepath, 'r', encoding="utf-8") as f:
            code = f.read()

        main_block=False
        if _has_main_block(self, code): main_block=True
        
        if main_block:
            namespace = {'__name__': '__main__', '__file__': filepath}
            exec(code, namespace)

            found_vars={}
            for var_name in circuit_vars:
                variable = namespace.get(var_name)
                found_vars[var_name]=variable
            
            circuit_vars=found_vars

        else:
            # Execute the file first to get the final circuits for reference
            spec = importlib.util.spec_from_file_location("quantum_module", filepath)
            module = importlib.util.module_from_spec(spec)

            
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                print(f"Error executing module: {e}")
                return results

            


        if debug: print("Circuit Vars: ",circuit_vars)


        # Get the final circuits and update sizes
        final_circuits = {}


        if main_block:
        
            for var_name in circuit_vars:   

                try:         

                    circuit = circuit_vars[var_name]

                    if (hasattr(circuit, 'data') and 
                        (hasattr(circuit, 'qubits') or hasattr(circuit, 'num_qubits'))):
                        final_circuits[var_name] = circuit
                        # Update circuit size from actual circuit
                        if hasattr(circuit, 'num_qubits'):
                            self.circuit_sizes[var_name] = circuit.num_qubits
                        elif hasattr(circuit, 'qubits'):
                            self.circuit_sizes[var_name] = len(circuit.qubits)
                    
                except: continue
        
        else:
            for var_name in circuit_vars:

                try:

                    if hasattr(module, var_name):
                        circuit = getattr(module, var_name)
                        if (hasattr(circuit, 'data') and 
                            (hasattr(circuit, 'qubits') or hasattr(circuit, 'num_qubits'))):
                            final_circuits[var_name] = circuit
                            # Update circuit size from actual circuit
                            if hasattr(circuit, 'num_qubits'):
                                self.circuit_sizes[var_name] = circuit.num_qubits
                            elif hasattr(circuit, 'qubits'):
                                self.circuit_sizes[var_name] = len(circuit.qubits)
                except: continue

        
        



        """
        # Get the final circuits and update sizes
        final_circuits = {}
        for var_name in circuit_vars:
            if hasattr(module, var_name):
                circuit = getattr(module, var_name)
                if (hasattr(circuit, 'data') and 
                    (hasattr(circuit, 'qubits') or hasattr(circuit, 'num_qubits'))):
                    final_circuits[var_name] = circuit
                    # Update circuit size from actual circuit
                    if hasattr(circuit, 'num_qubits'):
                        self.circuit_sizes[var_name] = circuit.num_qubits
                    elif hasattr(circuit, 'qubits'):
                        self.circuit_sizes[var_name] = len(circuit.qubits)
        """
            

    

        if debug: print(f"Final circuits: {final_circuits} ")
        
        # Initialize subcircuit states - start empty, will be built dynamically
        for var_name in circuit_vars:
            subcircuit_snapshots[var_name] = []
        
        # Now simulate the execution step by step
        operation_lines = self._extract_operation_sequence(cleaned_source_code, circuit_vars)

        
        if debug:
            print(f"Found {len(operation_lines)} operation lines")
            for i, (line_num, line, op_type, details) in enumerate(operation_lines):
                print(f"  {i}: Line {line_num + 1}: {line.strip()} -> {op_type} -> {details}")
            
            print(f"\nSubcircuit snapshots:")
            for name, ops in subcircuit_snapshots.items():
                print(f"  {name}: {len(ops)} operations")
                for op in ops:
                    print(f"    - {op}")
        
        # Process each operation in sequence
        for line_num, line, op_type, details in operation_lines:
            if debug:
                print(f"Processing: Line {line_num + 1}: {line.strip()} -> {op_type}")
                if op_type == 'append':
                    subcircuit_name = details['subcircuit']
                    print(f"  Subcircuit '{subcircuit_name}' has {len(subcircuit_snapshots.get(subcircuit_name, []))} operations")
            
            if op_type == 'append':
                # Handle append operations
                main_circuit = details['main_circuit']
                subcircuit_name = details['subcircuit']
                qubits = details.get('qubits', [])
                clbits = details.get('clbits', [])
                
                # Get current state of the subcircuit
                current_subcircuit_ops = subcircuit_snapshots.get(subcircuit_name, [])
                
                if debug:
                    print(f"Appending {subcircuit_name} (with {len(current_subcircuit_ops)} ops) to {main_circuit} on qubits {qubits}, clbits {clbits}")
                    for op in current_subcircuit_ops:
                        print(f"    - {op}")

                # Add operations from the current state of the subcircuit
                for sub_op in current_subcircuit_ops:
                    # Map subcircuit qubits to main circuit qubits
                    mapped_qubits = self._map_qubits(sub_op['qubits_affected'], qubits)
                    
                    # Map subcircuit clbits to main circuit clbits (if any)
                    mapped_clbits = self._map_clbits(sub_op.get('clbits_affected', []), clbits)
                    
                    # Get proper column positions from the actual source line
                    actual_line = lines[line_num] if line_num < len(lines) else ''
                    column_start, column_end = self._get_actual_column_positions(actual_line, details.get('operation_pattern', ''))
                    
                    operation_info = {
                        'operation_name': sub_op['operation_name'],
                        'qubits_affected': mapped_qubits,
                        'clbits_affected': mapped_clbits,
                        'row': line_num + 1,
                        'column_start': column_start,
                        'column_end': column_end,
                        'source_line': actual_line.strip()
                    }
                    
                    # Only include params if they are actual qiskit Parameters, not circuit names
                    # Also exclude params for operations that don't use qiskit Parameters
                    if 'params' in sub_op and sub_op['params'] and not self._is_operation_without_params(sub_op['operation_name']):
                        filtered_params = self._filter_actual_parameters(sub_op['params'], circuit_vars)
                        if filtered_params:
                            operation_info['params'] = filtered_params
                    
                    results[main_circuit].append(operation_info)
                
            elif op_type == 'direct_operation':
                # Handle direct operations on circuits
                circuit_name = details['circuit']
                operation_name = details['operation']
                qubits = details.get('qubits', [])
                clbits = details.get('clbits', [])
                params = details.get('params', [])
                
                # Handle operations that affect all qubits when no specific qubits are given
                if not qubits and self._is_all_qubit_operation(operation_name):
                    circuit_size = self.circuit_sizes.get(circuit_name, 0)
                    qubits = list(range(circuit_size))
                
                # Get proper column positions from the actual source line
                actual_line = lines[line_num] if line_num < len(lines) else ''
                column_start, column_end = self._get_actual_column_positions(actual_line, details.get('operation_pattern', ''))
                
                operation_info = {
                    'operation_name': operation_name,
                    'qubits_affected': qubits,
                    'clbits_affected': clbits,
                    'row': line_num + 1,
                    'column_start': column_start,
                    'column_end': column_end,
                    'source_line': actual_line.strip()
                }
                
                # Only include params if they are actual qiskit Parameters, not circuit names
                # Also exclude params for operations that don't use qiskit Parameters
                if params and not self._is_operation_without_params(operation_name):
                    filtered_params = self._filter_actual_parameters(params, circuit_vars)
                    if filtered_params:
                        operation_info['params'] = filtered_params
                
                # DYNAMIC TRACKING: If this operation is on a subcircuit, add it to the subcircuit's state
                if circuit_name in subcircuit_snapshots:
                    subcircuit_op_info = {
                        'operation_name': operation_name,
                        'qubits_affected': qubits,
                        'clbits_affected': clbits
                    }
                    
                    # Add params to subcircuit operation info too
                    if params and not self._is_operation_without_params(operation_name):
                        filtered_params = self._filter_actual_parameters(params, circuit_vars)
                        if filtered_params:
                            subcircuit_op_info['params'] = filtered_params
                    
                    subcircuit_snapshots[circuit_name].append(subcircuit_op_info)
                    
                    if debug:
                        print(f"  Added operation to subcircuit '{circuit_name}': {subcircuit_op_info}")
                        print(f"  Subcircuit '{circuit_name}' now has {len(subcircuit_snapshots[circuit_name])} operations")
                
                # Add to the appropriate circuit - but only if it's the main circuit or if we're tracking all operations
                if circuit_name in results:
                    results[circuit_name].append(operation_info)
        
        return results
    
    def _is_all_qubit_operation(self, operation_name: str) -> bool:
        """Check if an operation affects all qubits when no specific qubits are provided."""
        all_qubit_operations = {
            'barrier', 'reset', 'measure_all', 'remove_final_measurements',
            'reverse_bits', 'clear', 'save_state', 'save_density_matrix',
            'save_probabilities', 'save_probabilities_dict', 'save_amplitudes',
            'save_amplitudes_squared', 'save_expectation_value', 'save_statevector',
            'repeat'  # Added repeat operation
        }
        return operation_name.lower() in all_qubit_operations
    
    def _get_actual_column_positions(self, line: str, pattern: str) -> Tuple[int, int]:
        """Get the actual column start and end positions from the source line."""
        if not line.strip() or not pattern:
            return 0, len(line)
        
        # Find the first non-whitespace character (start of the operation)
        column_start = len(line) - len(line.lstrip())
        
        # Try to find the end of the operation
        stripped_line = line.strip()
        if stripped_line:
            # Find where the actual operation ends in the original line
            operation_start_in_stripped = 0  # Operation starts at beginning of stripped line
            operation_end_in_stripped = len(stripped_line)
            
            # Look for common operation patterns to find the actual end
            match = re.search(r'(\w+\.[^(]+\([^)]*\))', stripped_line)
            if match:
                operation_end_in_stripped = match.end()
            
            column_end = column_start + operation_end_in_stripped
        else:
            column_end = len(line)
        
        return column_start, column_end
    
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
        """Process lines and expand loops to track execution sequence, handling nested loops."""
        operations = []
        expanded_lines = self._expand_nested_loops(lines, {})
        
        for line_info in expanded_lines:
            line_num, expanded_line, variable_context = line_info
            op_info = self._parse_operation_line(expanded_line, line_num, circuit_vars)
            if op_info:
                operations.append(op_info)
        
        return operations
    
    def _expand_nested_loops(self, lines: List[str], outer_vars: Dict[str, int]) -> List[Tuple[int, str, Dict]]:
        """Recursively expand nested loops and return list of (line_num, expanded_line, variable_context)."""
        expanded_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()
            
            if not stripped_line or stripped_line.startswith('#'):
                i += 1
                continue
            
            # Check for for loops - FIXED: Better regex pattern
            for_match = re.match(r'for\s+(\w+)\s+in\s+range\s*\(\s*(\d+)\s*\)\s*:', stripped_line)
            if for_match:
                loop_var = for_match.group(1)
                loop_count = int(for_match.group(2))
                
                # Find the loop body
                loop_body_start = i + 1
                loop_body_end = self._find_loop_end(lines, i)
                loop_body = lines[loop_body_start:loop_body_end]
                
                # Expand the loop for each iteration
                for loop_iteration in range(loop_count):
                    # Create new variable context including this loop variable
                    current_vars = outer_vars.copy()
                    current_vars[loop_var] = loop_iteration
                    
                    # Recursively process the loop body (handles nested loops)
                    nested_expanded = self._expand_nested_loops(loop_body, current_vars)
                    
                    # Add all expanded lines from this iteration
                    for nested_line_offset, nested_expanded_line, nested_context in nested_expanded:
                        actual_line_num = loop_body_start + nested_line_offset
                        expanded_lines.append((actual_line_num, nested_expanded_line, current_vars))
                
                i = loop_body_end
                continue
            
            # Process regular lines (not loop headers)
            if outer_vars:
                # Apply variable substitutions from all outer loops
                expanded_line = line
                for var_name, var_value in outer_vars.items():
                    expanded_line = self._expand_loop_variables(expanded_line, var_name, var_value)
                expanded_lines.append((i, expanded_line, outer_vars))
            else:
                # No loop variables to substitute
                expanded_lines.append((i, line, {}))
            
            i += 1
        
        return expanded_lines
    
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
        
        # Skip empty lines and comments
        if not line_stripped or line_stripped.startswith('#'):
            return None
        
        for circuit_name in circuit_vars:
            if circuit_name in line_stripped:
                # Check for append operations - IMPROVED REGEX to handle various append formats
                # This handles: circuit.append(subcircuit, [qubits]) and circuit.append(subcircuit, [qubits], [clbits])
                append_pattern = rf'{re.escape(circuit_name)}\.append\s*\(\s*(\w+)\s*,\s*\[([^\]]*)\](?:\s*,\s*\[([^\]]*)\])?\s*\)'
                append_match = re.search(append_pattern, line_stripped)
                if append_match:
                    subcircuit_name = append_match.group(1)
                    qubits_str = append_match.group(2)
                    clbits_str = append_match.group(3) if append_match.group(3) is not None else ""
                    
                    # Parse qubits - handle both numbers and variables (which should already be expanded)
                    qubits = []
                    if qubits_str.strip():
                        for q in qubits_str.split(','):
                            q = q.strip()
                            if q.isdigit():
                                qubits.append(int(q))
                            else:
                                # Try to evaluate as a simple expression or number
                                try:
                                    result = int(eval(q, {"__builtins__": {}}, {}))
                                    qubits.append(result)
                                except:
                                    print(f"Warning: Could not parse qubit '{q}' in line: {line_stripped}")
                                    qubits.append(0)  # Default fallback
                    
                    # Parse clbits - handle both numbers and variables (which should already be expanded)
                    clbits = []
                    if clbits_str.strip():
                        for c in clbits_str.split(','):
                            c = c.strip()
                            if c.isdigit():
                                clbits.append(int(c))
                            else:
                                # Try to evaluate as a simple expression or number
                                try:
                                    result = int(eval(c, {"__builtins__": {}}, {}))
                                    clbits.append(result)
                                except:
                                    print(f"Warning: Could not parse clbit '{c}' in line: {line_stripped}")
                                    clbits.append(0)  # Default fallback
                    
                    return (line_num, line, 'append', {
                        'main_circuit': circuit_name,
                        'subcircuit': subcircuit_name,
                        'qubits': qubits,
                        'clbits': clbits,
                        'operation_pattern': append_match.group(0)
                    })
                
                # Check for direct operations
                op_match = re.search(rf'{re.escape(circuit_name)}\.(\w+)\s*\(([^)]*)\)', line_stripped)
                if op_match:
                    operation_name = op_match.group(1)
                    params_str = op_match.group(2)
                    
                    # Parse parameters and qubits
                    qubits, clbits, params = self._parse_operation_params(params_str, operation_name)
                    
                    return (line_num, line, 'direct_operation', {
                        'circuit': circuit_name,
                        'operation': operation_name,
                        'qubits': qubits,
                        'clbits': clbits,
                        'params': params,
                        'operation_pattern': op_match.group(0)
                    })
        
        return None
    
    def _parse_operation_params(self, params_str: str, operation_name: str = None) -> Tuple[List[int], List[int], List]:
        """Parse operation parameters to extract qubits, clbits, and other parameters."""
        if not params_str.strip():
            return [], [], []
        
        qubits = []
        clbits = []
        params = []
        
        # For operations that affect all qubits, don't treat numeric parameters as qubits
        if operation_name and self._is_all_qubit_operation(operation_name):
            # For all-qubit operations, all parameters are just parameters, not qubits
            parts = [p.strip() for p in params_str.split(',')]
            for part in parts:
                if part and not part.isspace():
                    params.append(part)
            return [], [], params
        
        # Special handling for measure operation - FIXED
        if operation_name and operation_name.lower() == 'measure':
            return self._parse_measure_params(params_str)
        
        # Split parameters
        parts = [p.strip() for p in params_str.split(',')]
        
        for part in parts:
            # Check if this is a quantum register (affects all qubits in the register)
            if part in self.register_info and self.register_info[part]['type'] == 'quantum':
                register_size = self.register_info[part]['size']
                qubits.extend(list(range(register_size)))
            # Check if this is a classical register (affects all clbits in the register)
            elif part in self.register_info and self.register_info[part]['type'] == 'classical':
                register_size = self.register_info[part]['size']
                clbits.extend(list(range(register_size)))
            # Check for qubit specifications with brackets
            elif '[' in part and ']' in part:
                # Extract numbers from brackets
                bracket_content = re.search(r'\[([^\]]+)\]', part)
                if bracket_content:
                    nums = [int(q.strip()) for q in bracket_content.group(1).split(',') if q.strip().isdigit()]
                    # Determine if this is a register access or just a list
                    register_name = part.split('[')[0].strip()
                    if register_name in self.register_info:
                        if self.register_info[register_name]['type'] == 'quantum':
                            qubits.extend(nums)
                        elif self.register_info[register_name]['type'] == 'classical':
                            clbits.extend(nums)
                    else:
                        qubits.extend(nums)
            elif part.isdigit():
                qubits.append(int(part))
            else:
                # This might be a parameter (like phi) - only add if it's not a register name
                if part and not part.isspace() and part not in self.register_info:
                    params.append(part)
        
        return qubits, clbits, params
    
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
    
    def _map_clbits(self, subcircuit_clbits: List[int], target_clbits: List[int]) -> List[int]:
        """Map subcircuit clbit indices to main circuit clbit indices."""
        if not subcircuit_clbits:
            return target_clbits
        
        mapped = []
        for sub_clbit in subcircuit_clbits:
            if sub_clbit < len(target_clbits):
                mapped.append(target_clbits[sub_clbit])
        
        return mapped if mapped else target_clbits
    
    def _filter_actual_parameters(self, params: List[str], circuit_vars: List[str]) -> List[str]:
        """Filter out circuit names and register names, keep only actual qiskit Parameters."""
        filtered = []
        for param in params:
            # Skip if it's a circuit variable name or register name
            if param not in circuit_vars and param not in self.register_info:
                # This is a simplified check - in a real implementation, you might want to
                # check if it's actually a qiskit.circuit.Parameter instance
                # For now, we assume anything that's not a circuit name or register name could be a parameter
                filtered.append(param)
        return filtered
    
    def _is_operation_without_params(self, operation_name: str) -> bool:
        """Check if an operation doesn't use qiskit Parameters (like repeat, barrier)."""
        operations_without_params = {
            'repeat', 'barrier', 'measure', 'reset', 'measure_all',
            'remove_final_measurements', 'reverse_bits', 'clear'
        }
        return operation_name.lower() in operations_without_params
    
    def _expand_loop_variables(self, line: str, loop_var: str, loop_iteration: int) -> str:
        """Expand loop variables and evaluate mathematical expressions - FIXED to avoid replacing inside variable names."""
        expanded_line = line
        
        # Replace loop variable in brackets and parentheses FIRST (most specific)
        expanded_line = expanded_line.replace(f'[{loop_var}]', f'[{loop_iteration}]')
        expanded_line = expanded_line.replace(f'({loop_var})', f'({loop_iteration})')
        
        # Find and evaluate mathematical expressions containing the loop variable
        # Look for patterns like: operation(expression*loop_var, other_args)
        pattern = rf'([^,\(\s]+\*{re.escape(loop_var)}|{re.escape(loop_var)}\*[^,\)\s]+|\d+\.?\d*\*{re.escape(loop_var)}|{re.escape(loop_var)}\*\d+\.?\d*)'
        
        def evaluate_expression(match):
            expr = match.group(0)
            try:
                # Replace the loop variable with its current value
                expr_with_value = expr.replace(loop_var, str(loop_iteration))
                # Safely evaluate the mathematical expression
                result = eval(expr_with_value, {"__builtins__": {}}, {})
                return str(result)
            except:
                # If evaluation fails, return the original expression with variable substituted
                return expr.replace(loop_var, str(loop_iteration))
        
        expanded_line = re.sub(pattern, evaluate_expression, expanded_line)
        
        # FIXED: Only replace standalone loop variables, not those inside other words
        # Use word boundaries to avoid replacing parts of variable names
        standalone_pattern = rf'\b{re.escape(loop_var)}\b'
        expanded_line = re.sub(standalone_pattern, str(loop_iteration), expanded_line)
        
        return expanded_line
    
    def _parse_measure_params(self, params_str: str) -> Tuple[List[int], List[int], List]:
        """Special parsing for measure operation parameters (qubit, cbit). FIXED VERSION."""
        if not params_str.strip():
            return [], [], []
        
        qubits = []
        clbits = []
        params = []
        
        # Split parameters
        parts = [p.strip() for p in params_str.split(',')]
        
        if len(parts) >= 2:
            # First parameter is qubit
            qubit_part = parts[0]
            if qubit_part.isdigit():
                qubits.append(int(qubit_part))
            elif '[' in qubit_part and ']' in qubit_part:
                bracket_content = re.search(r'\[([^\]]+)\]', qubit_part)
                if bracket_content:
                    qubit_nums = [int(q.strip()) for q in bracket_content.group(1).split(',') if q.strip().isdigit()]
                    qubits.extend(qubit_nums)
            elif qubit_part in self.register_info and self.register_info[qubit_part]['type'] == 'quantum':
                # Handle quantum register
                register_size = self.register_info[qubit_part]['size']
                qubits.extend(list(range(register_size)))
            
            # Second parameter is classical bit
            clbit_part = parts[1]
            if clbit_part.isdigit():
                clbits.append(int(clbit_part))
            elif '[' in clbit_part and ']' in clbit_part:
                # Extract the index from creg[index] pattern
                bracket_content = re.search(r'\[([^\]]+)\]', clbit_part)
                if bracket_content:
                    clbit_nums = [int(c.strip()) for c in bracket_content.group(1).split(',') if c.strip().isdigit()]
                    clbits.extend(clbit_nums)
            elif clbit_part in self.register_info and self.register_info[clbit_part]['type'] == 'classical':
                # Handle classical register
                register_size = self.register_info[clbit_part]['size']
                clbits.extend(list(range(register_size)))
            
            # Any additional parameters are actual parameters
            if len(parts) > 2:
                for part in parts[2:]:
                    if part and not part.isspace():
                        params.append(part)
        
        elif len(parts) == 1:
            # Only one parameter - treat as qubit
            qubit_part = parts[0]
            if qubit_part.isdigit():
                qubits.append(int(qubit_part))
            elif '[' in qubit_part and ']' in qubit_part:
                bracket_content = re.search(r'\[([^\]]+)\]', qubit_part)
                if bracket_content:
                    qubit_nums = [int(q.strip()) for q in bracket_content.group(1).split(',') if q.strip().isdigit()]
                    qubits.extend(qubit_nums)
            elif qubit_part in self.register_info and self.register_info[qubit_part]['type'] == 'quantum':
                # Handle quantum register
                register_size = self.register_info[qubit_part]['size']
                qubits.extend(list(range(register_size)))
        
        return qubits, clbits, params


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

    return results