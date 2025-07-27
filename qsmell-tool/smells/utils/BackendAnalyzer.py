import ast
import re
import sys
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import importlib.util
import copy

"""
Backend and Run Execution Analyzer for Qiskit code
Tracks backend instances and run function executions by analyzing Python files
"""

class BackendAnalyzer:
    def __init__(self):
        self.backend_instances = {}  # Store backend variable names and their info
        self.run_executions = []  # Store run function executions
        self.circuit_variables = []  # Track circuit variables
        self.backend_variables = []  # Track backend variables
        self.circuit_instances = {}  # Store actual QuantumCircuit instances
        
    def analyze_file(self, filepath: str, debug: bool = False) -> Tuple[Dict[str, Any], Dict[str, Dict], List[Dict]]:
        """
        Analyze a Python file to extract circuit instances, backend instances and run executions.
        
        Args:
            filepath: Path to the Python file to analyze
            debug: If True, print debugging information
            
        Returns:
            Tuple of (circuit_instances, backend_instances, run_executions)
        """
        # Read the source code
        with open(filepath, 'r') as f:
            source_code = f.read()
        
        if debug:
            print(f"Analyzing file: {filepath}")
        
        # Parse the AST
        tree = ast.parse(source_code)
        
        # Find backend and circuit variables
        self._find_backend_variables(tree, source_code, debug)
        self._find_circuit_variables(tree, source_code, debug)
        
        if debug:
            print(f"Found backend variables: {self.backend_variables}")
            print(f"Found circuit variables: {self.circuit_variables}")
        
        # Execute the file to get actual backend instances and circuits
        self._execute_and_extract_backends_and_circuits(filepath, debug)
        
        # Analyze run executions in the source code
        self._analyze_run_executions(source_code, debug)
        
        return self.circuit_instances, self.backend_instances, self.run_executions
    
    def _find_backend_variables(self, tree: ast.AST, source_code: str, debug: bool = False):
        """Find all backend variable assignments in the AST."""
        lines = source_code.split('\n')
        
        # Common backend patterns to look for
        backend_patterns = [
            'Aer.get_backend',
            'AerSimulator',  
            'FakeProvider',
            'fake_',  # For fake backends like fake_manila
            'IBMQ.get_backend',
            'IBMProvider',
            'qiskit_ibm_runtime',
            'Runtime',
            'Backend',
            'Simulator',
            'get_backend'
        ]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Check if this looks like a backend assignment
                        line_num = node.lineno - 1
                        if line_num < len(lines):
                            line = lines[line_num]
                            
                            # Check if any backend pattern appears in the line
                            for pattern in backend_patterns:
                                if pattern in line:
                                    backend_info = {
                                        'variable_name': target.id,
                                        'line_number': node.lineno,
                                        'source_line': line.strip(),
                                        'backend_type': self._extract_backend_type(line)
                                    }
                                    self.backend_variables.append(target.id)
                                    self.backend_instances[target.id] = backend_info
                                    
                                    if debug:
                                        print(f"Found backend variable: {target.id} at line {node.lineno}")
                                    break
    
    def _find_circuit_variables(self, tree: ast.AST, source_code: str, debug: bool = False):
        """Find all QuantumCircuit variables in the AST."""
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
                                self.circuit_variables.append(target.id)
                                if debug:
                                    print(f"Found circuit variable: {target.id}")
    
    def _extract_backend_type(self, line: str) -> str:
        """Extract the backend type from the source line."""
        line_lower = line.lower()
        
        if 'aersimulator' in line_lower:
            return 'AerSimulator'
        elif 'aer.get_backend' in line_lower:
            # Try to extract the backend name from quotes
            match = re.search(r'get_backend\s*\(\s*[\'"]([^\'"]+)[\'"]', line)
            if match:
                return f"Aer_{match.group(1)}"
            return 'Aer_backend'
        elif 'fake_' in line_lower:
            # Extract fake backend name
            match = re.search(r'(fake_\w+)', line_lower)
            if match:
                return match.group(1)
            return 'FakeBackend'
        elif 'ibmq.get_backend' in line_lower or 'ibmprovider' in line_lower:
            return 'IBM_backend'
        elif 'runtime' in line_lower:
            return 'Runtime_backend'
        else:
            return 'Unknown_backend'
    
    def _execute_and_extract_backends_and_circuits(self, filepath: str, debug: bool = False):
        """Execute the file and extract actual backend instances and circuit instances."""
        spec = importlib.util.spec_from_file_location("backend_module", filepath)
        module = importlib.util.module_from_spec(spec)
        
        try:
            # Execute the module
            spec.loader.exec_module(module)
            
            # Extract circuit instances from the executed module
            for circuit_var in self.circuit_variables:
                if hasattr(module, circuit_var):
                    circuit_instance = getattr(module, circuit_var)
                    
                    # Check if it's actually a QuantumCircuit instance
                    if hasattr(circuit_instance, 'data') and (hasattr(circuit_instance, 'qubits') or hasattr(circuit_instance, 'num_qubits')):
                        self.circuit_instances[circuit_var] = circuit_instance
                        
                        if debug:
                            print(f"Extracted circuit {circuit_var}: {circuit_instance.num_qubits} qubits, {len(circuit_instance.data)} operations")
            
            # Extract backend instances from the executed module
            for backend_var in self.backend_variables:
                if hasattr(module, backend_var):
                    backend_instance = getattr(module, backend_var)
                    
                    # Update backend info with actual instance details
                    if backend_var in self.backend_instances:
                        self.backend_instances[backend_var].update({
                            'instance': backend_instance,
                            'instance_type': type(backend_instance).__name__,
                            'module': type(backend_instance).__module__,
                            'has_run_method': hasattr(backend_instance, 'run')
                        })
                        
                        # Try to get backend-specific properties
                        if hasattr(backend_instance, 'name'):
                            try:
                                self.backend_instances[backend_var]['name'] = backend_instance.name()
                            except:
                                self.backend_instances[backend_var]['name'] = str(backend_instance)
                        
                        if hasattr(backend_instance, 'configuration'):
                            try:
                                config = backend_instance.configuration()
                                self.backend_instances[backend_var]['configuration'] = {
                                    'n_qubits': getattr(config, 'n_qubits', None),
                                    'simulator': getattr(config, 'simulator', None),
                                    'local': getattr(config, 'local', None)
                                }
                            except:
                                pass
                        
                        if debug:
                            print(f"Updated backend {backend_var} with instance info")
                            
        except Exception as e:
            if debug:
                print(f"Error executing module: {e}")
    
    def _analyze_run_executions(self, source_code: str, debug: bool = False):
        """Analyze the source code to find run method executions."""
        lines = source_code.split('\n')
        
        # Remove comments and strings for cleaner analysis
        cleaned_source = self._remove_comments_and_strings(source_code)
        cleaned_lines = cleaned_source.split('\n')
        
        # Expand loops to track execution sequence
        expanded_lines = self._expand_loops_for_runs(cleaned_lines)
        
        for line_info in expanded_lines:
            line_num, expanded_line, variable_context = line_info
            
            # Look for .run() method calls
            run_matches = re.finditer(r'(\w+)\.run\s*\(([^)]*)\)', expanded_line)
            
            for run_match in run_matches:
                backend_var = run_match.group(1)
                params_str = run_match.group(2)
                
                # Check if this is one of our tracked backends
                if backend_var in self.backend_variables:
                    # Parse the run parameters
                    circuits_used, other_params = self._parse_run_parameters(params_str)
                    
                    # Get the actual source line for reference
                    actual_line = lines[line_num] if line_num < len(lines) else expanded_line
                    
                    run_execution = {
                        'backend_variable': backend_var,
                        'circuits_used': circuits_used,
                        'parameters': other_params,
                        'line_number': line_num + 1,
                        'source_line': actual_line.strip(),
                        'variable_context': variable_context.copy() if variable_context else {}
                    }
                    
                    self.run_executions.append(run_execution)
                    
                    if debug:
                        print(f"Found run execution: {backend_var}.run() at line {line_num + 1}")
                        print(f"  Circuits: {circuits_used}")
                        print(f"  Parameters: {other_params}")
    
    def _parse_run_parameters(self, params_str: str) -> Tuple[List[str], Dict[str, str]]:
        """Parse run method parameters to extract circuits and other parameters."""
        if not params_str.strip():
            return [], {}
        
        circuits_used = []
        other_params = {}
        
        # Split parameters by comma, but be careful with nested structures
        params = self._smart_split_parameters(params_str)
        
        for i, param in enumerate(params):
            param = param.strip()
            
            if '=' in param:
                # Named parameter
                key, value = param.split('=', 1)
                other_params[key.strip()] = value.strip()
            else:
                # Positional parameter - first one is usually circuits
                if i == 0:
                    # This should be the circuits parameter
                    if param in self.circuit_variables:
                        circuits_used.append(param)
                    elif '[' in param and ']' in param:
                        # List of circuits
                        circuit_list = re.findall(r'\w+', param)
                        circuits_used.extend([c for c in circuit_list if c in self.circuit_variables])
                    else:
                        # Might be a circuit variable we didn't catch
                        circuits_used.append(param)
                else:
                    # Other positional parameters
                    other_params[f'param_{i}'] = param
        
        return circuits_used, other_params
    
    def _smart_split_parameters(self, params_str: str) -> List[str]:
        """Split parameters by comma, respecting nested structures."""
        params = []
        current_param = ""
        paren_depth = 0
        bracket_depth = 0
        in_string = False
        string_char = None
        
        for char in params_str:
            if char in ['"', "'"] and not in_string:
                in_string = True
                string_char = char
            elif char == string_char and in_string:
                in_string = False
                string_char = None
            elif not in_string:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == '[':
                    bracket_depth += 1
                elif char == ']':
                    bracket_depth -= 1
                elif char == ',' and paren_depth == 0 and bracket_depth == 0:
                    params.append(current_param.strip())
                    current_param = ""
                    continue
            
            current_param += char
        
        if current_param.strip():
            params.append(current_param.strip())
        
        return params
    
    def _expand_loops_for_runs(self, lines: List[str]) -> List[Tuple[int, str, Dict]]:
        """Expand loops to track run executions in loops."""
        expanded_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped_line = line.strip()
            
            if not stripped_line or stripped_line.startswith('#'):
                i += 1
                continue
            
            # Check for for loops
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
                    current_vars = {loop_var: loop_iteration}
                    
                    # Process the loop body
                    for j, body_line in enumerate(loop_body):
                        # Expand loop variables in the line
                        expanded_line = self._expand_loop_variables(body_line, loop_var, loop_iteration)
                        actual_line_num = loop_body_start + j
                        expanded_lines.append((actual_line_num, expanded_line, current_vars))
                
                i = loop_body_end
                continue
            
            # Regular line (not in a loop)
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
    
    def _expand_loop_variables(self, line: str, loop_var: str, loop_iteration: int) -> str:
        """Expand loop variables in a line."""
        expanded_line = line
        
        # Replace loop variable with its current value
        # Use word boundaries to avoid replacing parts of variable names
        pattern = rf'\b{re.escape(loop_var)}\b'
        expanded_line = re.sub(pattern, str(loop_iteration), expanded_line)
        
        return expanded_line
    
    def _remove_comments_and_strings(self, source_code: str) -> str:
        """Remove comments and string literals from source code."""
        lines = source_code.split('\n')
        cleaned_lines = []
        
        in_triple_quote_single = False
        in_triple_quote_double = False
        
        for line in lines:
            cleaned_line = ""
            j = 0
            
            while j < len(line):
                # Check for triple quotes
                if j <= len(line) - 3:
                    three_chars = line[j:j+3]
                    
                    if three_chars == '"""':
                        if in_triple_quote_double:
                            in_triple_quote_double = False
                            j += 3
                            continue
                        elif not in_triple_quote_single:
                            in_triple_quote_double = True
                            j += 3
                            continue
                    
                    elif three_chars == "'''":
                        if in_triple_quote_single:
                            in_triple_quote_single = False
                            j += 3
                            continue
                        elif not in_triple_quote_double:
                            in_triple_quote_single = True
                            j += 3
                            continue
                
                # Skip if inside multi-line string
                if in_triple_quote_single or in_triple_quote_double:
                    j += 1
                    continue
                
                # Handle single-line comments
                if line[j] == '#':
                    # Check if we're inside a string
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
                    
                    if not in_string:
                        break  # Rest of line is a comment
                
                cleaned_line += line[j]
                j += 1
            
            cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)


def get_circuits_from_file(filepath: str, debug: bool = False) -> Dict[str, Any]:
    """
    Extract circuit instances from a Python file.
    
    Args:
        filepath: Path to the Python file to analyze
        debug: If True, print debugging information
        
    Returns:
        Dictionary mapping circuit variable names to their QuantumCircuit instances
    """
    analyzer = BackendAnalyzer()
    circuits, _, _ = analyzer.analyze_file(filepath, debug)
    return circuits


def get_backends_from_file(filepath: str, debug: bool = False) -> Dict[str, Dict]:
    """
    Extract backend instances from a Python file.
    
    Args:
        filepath: Path to the Python file to analyze
        debug: If True, print debugging information
        
    Returns:
        Dictionary mapping backend variable names to their information
    """
    analyzer = BackendAnalyzer()
    _, backends, _ = analyzer.analyze_file(filepath, debug)
    return backends


def get_runs_from_file(filepath: str, debug: bool = False) -> List[Dict]:
    """
    Extract run function executions from a Python file.
    
    Args:
        filepath: Path to the Python file to analyze
        debug: If True, print debugging information
        
    Returns:
        List of run execution information dictionaries
    """
    analyzer = BackendAnalyzer()
    _, _, runs = analyzer.analyze_file(filepath, debug)
    return runs


def analyze_circuits_backends_runs(filepath: str, debug: bool = False) -> Tuple[Dict[str, Any], Dict[str, Dict], List[Dict]]:
    """
    Analyze a Python file to extract circuit instances, backend instances and run executions.
    
    Args:
        filepath: Path to the Python file to analyze
        debug: If True, print debugging information
        
    Returns:
        Tuple of (circuit_instances, backend_instances, run_executions)
    """
    analyzer = BackendAnalyzer()
    return analyzer.analyze_file(filepath, debug)
