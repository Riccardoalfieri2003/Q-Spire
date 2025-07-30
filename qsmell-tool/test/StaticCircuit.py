import ast
import inspect
import importlib
import os
import sys
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import re


@dataclass
class CircuitInfo:
    """Information about a found QuantumCircuit."""
    name: str
    location: str  # 'parameter' or 'created'
    function_name: str
    line_number: int
    assignment_line: Optional[str] = None  # The actual line of code where circuit is created


@dataclass
class ParameterInfo:
    """Information about function parameters."""
    name: str
    type_annotation: str
    default_value: Any
    is_custom_type: bool


class FunctionExecutionGenerator:
    """
    Generates executable code from function bodies with proper parameter instantiation.
    """
    
    def __init__(self):
        self.circuits_found: List[CircuitInfo] = []
        self.functions_with_circuits: Dict[str, Dict[str, Any]] = {}
        self.imports: Set[str] = set()
        self.class_definitions: Dict[str, str] = {}
        self.module_functions: Dict[str, str] = {}  # Store module-level functions
        self.class_constructors: Dict[str, List[str]] = {}  # Store constructor signatures
    
    def analyze_and_generate_executable(self, file_path: str, function_name: str) -> str:
        """
        Analyze a file and generate executable code for a specific function.
        
        Args:
            file_path: Path to the Python file
            function_name: Name of the function to make executable
            
        Returns:
            Executable Python code as string
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return self.generate_executable_from_code(content, function_name, file_path)
    
    def analyze_and_generate_all_executables(self, file_path: str, output_dir: str = None) -> Dict[str, str]:
        """
        Analyze a file and generate executable code for ALL functions that contain QuantumCircuits.
        
        Args:
            file_path: Path to the Python file
            output_dir: Directory to save the executable files (optional)
            
        Returns:
            Dictionary mapping function names to their executable code
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Find all functions with QuantumCircuits
        functions_with_circuits = self.find_all_functions_with_circuits(content, file_path)
        
        # Generate executable code for each function
        executables = {}
        for func_name in functions_with_circuits:
            executable_code = self.generate_executable_from_code(content, func_name, file_path)
            executables[func_name] = executable_code
            
            # Save to file if output directory is specified
            if output_dir:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                output_file = os.path.join(output_dir, f"executable_{func_name}.py")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(executable_code.replace("nonlocal ",""))
                #print(f"Generated executable for function '{func_name}': {output_file}")
        
        return executables
    
    def find_all_functions_with_circuits(self, code: str, source_name: str = "<string>") -> List[str]:
        """
        Find all functions that contain QuantumCircuits (either as parameters or created in body).
        
        Args:
            code: Python code to analyze
            source_name: Source identifier
            
        Returns:
            List of function names that contain QuantumCircuits
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            print(f"Syntax error in {source_name}: {e}")
            return []
        
        # Reset state
        self.circuits_found = []
        self.functions_with_circuits = {}
        self.imports = set()
        self.class_definitions = {}
        self.module_functions = {}
        self.class_constructors = {}
        self.code_lines = code.split('\n')
        
        # Extract imports, class definitions, and constructor signatures
        self._extract_imports_and_classes(tree)
        
        # Find all functions and analyze them
        functions_with_circuits = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Analyze each function
                self._analyze_function(node)
                
                # Check if this function has any circuits
                if node.name in self.functions_with_circuits:
                    func_info = self.functions_with_circuits[node.name]
                    if func_info['circuits']:  # Has at least one circuit
                        functions_with_circuits.append(node.name)
        
        return functions_with_circuits
    
    def generate_executable_from_code(self, code: str, function_name: str, source_name: str = "<string>") -> str:
        """
        Generate executable code from a function in the given code.
        
        Args:
            code: Python code containing the function
            function_name: Name of the function to extract
            source_name: Source identifier
            
        Returns:
            Executable Python code as string
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"# Syntax error in {source_name}: {e}"
        
        # Reset state
        self.circuits_found = []
        self.functions_with_circuits = {}
        self.imports = set()
        self.class_definitions = {}
        self.module_functions = {}
        self.class_constructors = {}
        self.code_lines = code.split('\n')
        
        # Extract imports, class definitions, and constructor signatures
        self._extract_imports_and_classes(tree)
        
        # Find the target function
        target_function = self._find_function(tree, function_name)
        if not target_function:
            return f"# Function '{function_name}' not found in {source_name}"
        
        # Analyze the function
        self._analyze_function(target_function)
        
        # Generate executable code
        return self._generate_main_code(target_function, source_name)
    
    def _extract_imports_and_classes(self, tree: ast.AST):
        """Extract all imports, class definitions, constructor signatures, and module-level functions from the AST."""
        self.module_functions = {}  # Store module-level functions
        self.class_constructors = {}  # Store constructor parameter info
        
        # Process top-level nodes only
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_name = alias.name
                    if alias.asname:
                        import_name += f" as {alias.asname}"
                    self.imports.add(f"import {import_name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module or node.level > 0:
                    # Handle relative imports by converting them to absolute
                    names = []
                    for alias in node.names:
                        if alias.asname:
                            names.append(f"{alias.name} as {alias.asname}")
                        else:
                            names.append(alias.name)
                    names_str = ", ".join(names)
                    
                    # Convert relative imports to absolute imports with try-except wrapper
                    if node.level > 0:  # Relative import
                        # Create a try-except import statement
                        original_import = f"from {'.' * node.level}{node.module or ''} import {names_str}"
                        absolute_import = self._guess_absolute_import(node, names_str)
                        
                        import_block = f"""try:
    {original_import}
except ImportError:
    try:
        {absolute_import}
    except ImportError:
        # Mock the imports if they can't be resolved
{self._create_mock_imports(names)}"""
                        
                        self.imports.add(import_block)
                    else:
                        # Regular absolute import
                        self.imports.add(f"from {node.module} import {names_str}")
            elif isinstance(node, ast.ClassDef):
                class_source = self._extract_node_source(node)
                self.class_definitions[node.name] = class_source
                
                # Extract constructor parameters
                constructor_params = self._extract_constructor_params(node)
                self.class_constructors[node.name] = constructor_params
                
            elif isinstance(node, ast.FunctionDef):
                # This is a module-level function
                func_source = self._extract_node_source(node)
                self.module_functions[node.name] = func_source
    
    def _extract_constructor_params(self, class_node: ast.ClassDef) -> List[str]:
        """Extract constructor parameter names from a class definition."""
        constructor_params = []
        
        # Find the __init__ method
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                # Extract parameter names (excluding 'self')
                for arg in node.args.args[1:]:  # Skip 'self'
                    constructor_params.append(arg.arg)
                break
        
        return constructor_params
    
    def _guess_absolute_import(self, import_node: ast.ImportFrom, names_str: str) -> str:
        """Guess the absolute import path for a relative import."""
        level = import_node.level
        module = import_node.module or ""
        
        # Common patterns for qiskit and similar packages
        if level == 1:  # from .module
            if "ae_utils" in module or "amplitude_estimator" in module or "estimation_problem" in module:
                return f"from qiskit_algorithms.amplitude_estimators.{module} import {names_str}"
            elif module == "":
                return f"from qiskit_algorithms.amplitude_estimators import {names_str}"
        elif level == 2:  # from ..module  
            if "exceptions" in module:
                return f"from qiskit_algorithms.exceptions import {names_str}"
        
        # Generic fallback
        return f"from package.{module} import {names_str}" if module else f"from package import {names_str}"
    
    def _create_mock_imports(self, names_list: list) -> str:
        """Create mock objects for imports that can't be resolved."""
        mock_lines = []
        for alias in names_list:
            # Handle "name as alias" format
            if ' as ' in alias:
                original_name, alias_name = alias.split(' as ')
                mock_lines.append(f"        {alias_name.strip()} = type('Mock{original_name.strip()}', (), {{}})  # Mock object")
            else:
                name = alias.strip()
                mock_lines.append(f"        {name} = type('Mock{name}', (), {{}})  # Mock object")
        return '\n'.join(mock_lines)
    
    def _find_function(self, tree: ast.AST, function_name: str) -> Optional[ast.FunctionDef]:
        """Find a specific function in the AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return node
        return None
    
    def _analyze_function(self, func_node: ast.FunctionDef):
        """Analyze a function for QuantumCircuit instances."""
        func_name = func_node.name
        
        # Check function parameters for QuantumCircuit types
        parameter_circuits = self._find_parameter_circuits(func_node)
        
        # Check function body for QuantumCircuit creations
        created_circuits = self._find_created_circuits(func_node)
        
        # Store function info
        all_circuits = parameter_circuits + created_circuits
        func_source = self._extract_function_source(func_node)
        
        self.functions_with_circuits[func_name] = {
            "function_name": func_name,
            "line_number": func_node.lineno,
            "circuits": all_circuits,
            "source_code": func_source,
            "parameters": self._extract_parameters(func_node),
            "returns_annotation": self._get_return_annotation(func_node),
            "body": self._extract_function_body(func_node)
        }
        
        self.circuits_found.extend(all_circuits)
    
    def _extract_parameters(self, func_node: ast.FunctionDef) -> List[ParameterInfo]:
        """Extract detailed parameter information."""
        parameters = []
        
        # Handle regular parameters
        for arg in func_node.args.args:
            type_annotation = ""
            is_custom_type = False
            
            if arg.annotation:
                type_annotation = self._annotation_to_string(arg.annotation)
                is_custom_type = self._is_custom_type(type_annotation)
            
            # Handle default values
            default_value = None
            defaults_offset = len(func_node.args.args) - len(func_node.args.defaults)
            arg_index = func_node.args.args.index(arg)
            
            if arg_index >= defaults_offset:
                default_index = arg_index - defaults_offset
                default_value = self._ast_to_value(func_node.args.defaults[default_index])
            
            param_info = ParameterInfo(
                name=arg.arg,
                type_annotation=type_annotation,
                default_value=default_value,
                is_custom_type=is_custom_type
            )
            parameters.append(param_info)
        
        # Handle **kwargs parameters
        if func_node.args.kwarg:
            kwarg_param = ParameterInfo(
                name=func_node.args.kwarg.arg,
                type_annotation="dict",
                default_value=None,
                is_custom_type=False
            )
            parameters.append(kwarg_param)
        
        return parameters
    
    def _extract_function_body(self, func_node: ast.FunctionDef) -> str:
        """Extract just the body of a function (without def line), replacing return statements with sys.exit(0)."""
        # Use AST to get the exact boundaries
        if hasattr(func_node, 'end_lineno') and func_node.end_lineno:
            # Use the end_lineno from AST (Python 3.8+)
            start_line = func_node.lineno
            end_line = func_node.end_lineno
            
            # Find the first line of the function body (after the def line and docstring)
            body_start = func_node.lineno  # def line
            
            # Skip docstring if present
            if (func_node.body and 
                isinstance(func_node.body[0], ast.Expr) and 
                isinstance(func_node.body[0].value, ast.Constant) and 
                isinstance(func_node.body[0].value.value, str)):
                # Has docstring, skip it
                if len(func_node.body) > 1:
                    body_start = func_node.body[1].lineno - 1
                else:
                    return "    pass  # Empty function body"
            else:
                # No docstring, start from first statement
                if func_node.body:
                    body_start = func_node.body[0].lineno - 1
                else:
                    return "    pass  # Empty function body"
            
            # Extract body lines using AST boundaries
            body_lines = self.code_lines[body_start:end_line]
        else:
            # Fallback to manual detection for older Python versions
            # Find the first line of the function body (after the def line and docstring)
            body_start = func_node.lineno  # def line
            
            # Skip docstring if present
            if (func_node.body and 
                isinstance(func_node.body[0], ast.Expr) and 
                isinstance(func_node.body[0].value, ast.Constant) and 
                isinstance(func_node.body[0].value.value, str)):
                # Has docstring, skip it
                if len(func_node.body) > 1:
                    body_start = func_node.body[1].lineno - 1
                else:
                    return "    pass  # Empty function body"
            else:
                # No docstring, start from first statement
                if func_node.body:
                    body_start = func_node.body[0].lineno - 1
                else:
                    return "    pass  # Empty function body"
            
            # Find end of function manually
            end_line = self._find_node_end_line(func_node, func_node.lineno - 1)
            
            # Extract body lines
            body_lines = self.code_lines[body_start:end_line]
        
        # Calculate base indentation from the first non-empty line
        base_indent = 0
        if body_lines:
            for line in body_lines:
                if line.strip():
                    base_indent = self._get_indentation(line)
                    break
        
        # Remove base indentation and replace return statements with sys.exit(0)
        processed_lines = []
        for line in body_lines:
            if line.strip():  # Non-empty line
                stripped_line = line.strip()
                
                # Get the current line's indentation relative to base
                current_indent = self._get_indentation(line)
                relative_indent = max(0, current_indent - base_indent)
                indent_str = ' ' * relative_indent
                
                # Replace return statements with sys.exit(0)
                if stripped_line.startswith('return '):
                    # Add a comment showing what was returned, then sys.exit(0)
                    returned_value = stripped_line[7:].strip()  # Remove 'return '
                    processed_lines.append(f'{indent_str}# Original function returned: {returned_value}')
                    processed_lines.append(f'{indent_str}sys.exit(0)')
                    continue
                elif stripped_line == 'return':
                    # Handle bare return statements
                    processed_lines.append(f'{indent_str}# Original function had bare return')
                    processed_lines.append(f'{indent_str}sys.exit(0)')
                    continue
                
                # Skip or modify super() calls
                if 'super().__init__()' in stripped_line or 'super().' in stripped_line:
                    processed_lines.append(f'{indent_str}# Removed super() call: {stripped_line}')
                    continue
                
                # Remove base indentation but preserve relative indentation
                if line.startswith(' ' * base_indent):
                    processed_lines.append(line[base_indent:])
                else:
                    # If line doesn't have expected base indentation, just strip and add relative indent
                    processed_lines.append(f'{indent_str}{stripped_line}')
            else:
                processed_lines.append('')  # Keep empty lines
        
        return '\n'.join(processed_lines)
    
    def _generate_main_code(self, func_node: ast.FunctionDef, source_name: str) -> str:
        """Generate the main executable code."""
        func_info = self.functions_with_circuits[func_node.name]
        
        code_parts = []
        
        # Handle __future__ imports first (they must be at the top)
        future_imports = []
        regular_imports = []
        
        for imp in sorted(self.imports):
            if imp.startswith('from __future__'):
                future_imports.append(imp)
            else:
                regular_imports.append(imp)
        
        # Add __future__ imports first
        if future_imports:
            for imp in future_imports:
                code_parts.append(imp)
            code_parts.append('')
        
        # Add sys.path modification to find local modules
        code_parts.append('# Add current directory and parent directory to Python path for local imports')
        code_parts.append('import sys')
        code_parts.append('import os')
        code_parts.append('')
        
        # Add path setup
        if source_name != "<string>" and os.path.isfile(source_name):
            source_dir = os.path.dirname(os.path.abspath(source_name))
            parent_dir = os.path.dirname(source_dir)
            grandparent_dir = os.path.dirname(parent_dir)
            
            code_parts.extend([
                f'source_dir = r"{source_dir}"',
                f'parent_dir = r"{parent_dir}"',
                f'grandparent_dir = r"{grandparent_dir}"',
                '',
                'sys.path.insert(0, source_dir)',
                'sys.path.insert(0, parent_dir)',
                'sys.path.insert(0, grandparent_dir)',
                ''
            ])
        else:
            code_parts.extend([
                'current_dir = os.path.dirname(os.path.abspath(__file__))',
                'parent_dir = os.path.dirname(current_dir)',
                'grandparent_dir = os.path.dirname(parent_dir)',
                'sys.path.insert(0, current_dir)',
                'sys.path.insert(0, parent_dir)',
                'sys.path.insert(0, grandparent_dir)',
                ''
            ])
        
        # Add error handling for imports
        code_parts.extend([
            '# Handle imports with error handling',
            'import warnings',
            'warnings.filterwarnings("ignore", category=DeprecationWarning)',
            ''
        ])
        
        # Add regular imports
        code_parts.append('# Required imports')
        for imp in regular_imports:
            if '\n' in imp:
                code_parts.extend(imp.split('\n'))
            else:
                code_parts.append(imp)
        
        # Add standard imports that might be needed
        standard_imports = [
            'from qiskit import QuantumCircuit, ClassicalRegister',
            'from qiskit.circuit.library import *',
            'import numpy as np'
        ]
        
        for imp in standard_imports:
            if imp not in self.imports:
                code_parts.append(imp)
        
        code_parts.append('')
        
        # Add enhanced mock classes with proper parameter handling - FIXED VERSION
        code_parts.append('# Enhanced mock classes with proper parameter handling')
        code_parts.append('''class MockParameter:
    def __init__(self, name="param"):
        self.name = name
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"MockParameter('{self.name}')"
    
    def __eq__(self, other):
        if isinstance(other, MockParameter):
            return self.name == other.name
        return False
    
    def __hash__(self):
        return hash(self.name)

class MockParameterVector:
    def __init__(self, name, length):
        self.name = name
        self.length = length
        self._params = [MockParameter(f"{name}[{i}]") for i in range(length)]
    
    def __getitem__(self, index):
        return self._params[index]
    
    def __iter__(self):
        return iter(self._params)
    
    def __len__(self):
        return self.length

class MockParameterView:
    """Mock for QuantumCircuit.parameters which returns a ParameterView-like object."""
    def __init__(self, params=None):
        self._params = params or []
        # Create a data attribute that contains the parameters
        self.data = self._params
    
    def __iter__(self):
        return iter(self._params)
    
    def __len__(self):
        return len(self._params)
    
    def __bool__(self):
        return len(self._params) > 0
    
    def __contains__(self, item):
        return item in self._params
    
    def index(self, item):
        # Try direct lookup first
        if item in self._params:
            return self._params.index(item)
        
        # If not found, try by name (for MockParameter objects)
        if hasattr(item, 'name'):
            for i, param in enumerate(self._params):
                if hasattr(param, 'name') and param.name == item.name:
                    return i
        
        # If still not found, try string representation
        item_str = str(item)
        for i, param in enumerate(self._params):
            if str(param) == item_str:
                return i
        
        # If really not found, raise ValueError like the real implementation
        raise ValueError(f"{item} is not in parameters list. Available parameters: {[str(p) for p in self._params]}")

class MockQuantumCircuit:
    """Enhanced mock QuantumCircuit with proper parameter handling - FIXED VERSION."""
    def __init__(self, num_qubits=2, num_clbits=0, name=None):
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self.name = name or "circuit"
        
        # Create mock qubits and clbits
        self.qubits = [f"qubit_{i}" for i in range(num_qubits)]
        self.clbits = [f"clbit_{i}" for i in range(num_clbits)]
        
        # Create mock parameters - start with some default parameters that match common usage
        self._param_list = [
            MockParameter(f"theta_{i}") for i in range(max(2, num_qubits))
        ]
        self.parameters = MockParameterView(self._param_list)
        
        # Other circuit attributes
        self.data = []  # Instructions
        self.global_phase = 0
    
    @property
    def num_parameters(self):
        """Return the number of parameters as an integer (CRITICAL FIX for numpy compatibility)."""
        # Ensure this returns an integer for numpy compatibility
        count = len(self._param_list)
        return int(count)  # Explicit conversion to int
    
    def add_parameter(self, param):
        """Add a parameter to the circuit."""
        if param not in self._param_list:
            self._param_list.append(param)
            self.parameters = MockParameterView(self._param_list)
        return param
    
    def assign_parameters(self, param_dict):
        """Mock parameter assignment."""
        new_circuit = MockQuantumCircuit(self.num_qubits, self.num_clbits, self.name)
        # Filter out assigned parameters
        remaining_params = [p for p in self._param_list if p not in param_dict]
        new_circuit._param_list = remaining_params
        new_circuit.parameters = MockParameterView(remaining_params)
        return new_circuit
    
    def bind_parameters(self, param_dict):
        """Alias for assign_parameters."""
        return self.assign_parameters(param_dict)
    
    def copy(self):
        """Create a copy of the circuit."""
        new_circuit = MockQuantumCircuit(self.num_qubits, self.num_clbits, self.name)
        new_circuit._param_list = self._param_list.copy()
        new_circuit.parameters = MockParameterView(new_circuit._param_list)
        return new_circuit
    
    def __getattr__(self, name):
        # Default method for any other attributes - return a lambda that returns self for chaining
        if name.startswith('__') and name.endswith('__'):
            # Don't mock dunder methods
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return lambda *args, **kwargs: self

# Override the standard QuantumCircuit with our mock
QuantumCircuit = MockQuantumCircuit

class MockBaseEstimator:
    def __init__(self, *args, **kwargs):
        pass
    
    def run(self, circuits, observables, parameter_values=None, **kwargs):
        # Return a mock result that has a 'result()' method
        class MockJob:
            def result(self):
                class MockResult:
                    def __init__(self):
                        # Create mock values based on the number of circuits
                        num_circuits = len(circuits) if hasattr(circuits, '__len__') else 1
                        self.values = [0.5 + 0.1 * i for i in range(num_circuits)]
                return MockResult()
        return MockJob()

class MockBaseEstimatorGradient:
    def __init__(self, estimator=None, options=None):
        self._estimator = estimator or MockBaseEstimator()
        self._options = options or {}
    
    def _get_local_options(self, options):
        return {**self._options, **options}

class MockOptions:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class MockSparsePauliOp:
    """Mock for SparsePauliOp observables."""
    def __init__(self, data=None, coeffs=None):
        self.data = data or ["II", "XX", "YY", "ZZ"]
        self.coeffs = coeffs or [1.0, 1.0, 1.0, 1.0]
        self.num_qubits = 2
    
    def __iter__(self):
        return iter(zip(self.data, self.coeffs))

# Mock the SparsePauliOp if it's used
try:
    from qiskit.quantum_info import SparsePauliOp
except ImportError:
    SparsePauliOp = MockSparsePauliOp''')
        
        code_parts.append('')
        
        # Add class definitions if any are needed
        if self.class_definitions:
            code_parts.append('# Class definitions (with mock parent classes)')
            for class_name, class_code in self.class_definitions.items():
                # Modify class definitions to inherit from mock classes if needed
                modified_class_code = self._modify_class_for_mocking(class_code, class_name)
                code_parts.append(modified_class_code)
                code_parts.append('')
        
        # Add module-level functions if any are needed
        if self.module_functions:
            code_parts.append('# Module-level functions')
            for func_name, func_code in self.module_functions.items():
                code_parts.append(func_code)
                code_parts.append('')
        
        # Generate parameter instantiation
        code_parts.append('# Parameter instantiation')
        code_parts.append('if __name__ == "__main__":')
        
        # Find the class that contains this function (if any)
        containing_class = self._find_containing_class(func_node)
        
        # Create instances for each parameter
        for param in func_info['parameters']:
            instance_code = self._generate_parameter_instance(param, containing_class)
            # Handle multi-line instance code
            if '\n' in instance_code:
                for line in instance_code.split('\n'):
                    if line.strip():
                        code_parts.append(f'    {line}')
                    else:
                        code_parts.append('')
            else:
                code_parts.append(f'    {instance_code}')

        code_parts.append('')
        code_parts.append('# Fix parameter consistency for gradient calculations')
        code_parts.append('# Ensure parameter_values match circuit parameters exactly')
        code_parts.append('    if "parameter_values" in locals() and "circuits" in locals():')
        code_parts.append('        # Make parameter_values consistent with actual circuit parameters')
        code_parts.append('        if isinstance(parameter_values, list):')
        code_parts.append('            for i, circuit in enumerate(circuits):')
        code_parts.append('                if hasattr(circuit, "num_parameters") and i < len(parameter_values):')
        code_parts.append('                    current_values = parameter_values[i]')
        code_parts.append('                    expected_params = circuit.num_parameters')
        code_parts.append('                    ')
        code_parts.append('                    # Convert scalar to list')
        code_parts.append('                    if not isinstance(current_values, (list, tuple, np.ndarray)):')
        code_parts.append('                        current_values = [current_values]')
        code_parts.append('                    ')
        code_parts.append('                    # Handle multi-dimensional arrays (e.g., [[0.1, 0.2], [0.3, 0.4]])')
        code_parts.append('                    if isinstance(current_values, (list, tuple, np.ndarray)) and len(current_values) > 0:')
        code_parts.append('                        # Flatten nested structures')
        code_parts.append('                        flat_values = []')
        code_parts.append('                        for val in current_values:')
        code_parts.append('                            if isinstance(val, (list, tuple, np.ndarray)):')
        code_parts.append('                                flat_values.extend(val)')
        code_parts.append('                            else:')
        code_parts.append('                                flat_values.append(val)')
        code_parts.append('                        current_values = flat_values')
        code_parts.append('                    ')
        code_parts.append('                    # Adjust length to match expected_params')
        code_parts.append('                    current_length = len(current_values)')
        code_parts.append('                    current_values = np.array(current_values).flatten()')
        code_parts.append('                    if current_values.ndim == 0:')
        code_parts.append('                        current_values = np.array([current_values])')
        code_parts.append('                    current_values = current_values.tolist()')
        code_parts.append('                    if current_length != expected_params:')
        code_parts.append('                        if current_length < expected_params:')
        code_parts.append('                            padding = expected_params - current_length')
        code_parts.append('                            parameter_values[i] = current_values + [0.0] * padding')
        code_parts.append('                        else:')
        code_parts.append('                            parameter_values[i] = current_values[:expected_params]')
                
        
        # Add special handling for common parameter patterns
        code_parts.append('    # Ensure parameters match circuits')
        code_parts.append('    if "parameters" in locals() and "circuits" in locals():')
        code_parts.append('        # Make sure parameters match the circuits')
        code_parts.append('        if isinstance(parameters, list) and len(parameters) > 0:')
        code_parts.append('            if parameters[0] is None:  # If we have None parameters')
        code_parts.append('                # Replace with actual parameters from circuits')
        code_parts.append('                parameters = []')
        code_parts.append('                for i, circuit in enumerate(circuits):')
        code_parts.append('                    if hasattr(circuit, "parameters") and circuit.parameters:')
        code_parts.append('                        # Take first 2 parameters from each circuit')
        code_parts.append('                        circuit_params = list(circuit.parameters)[:2]')
        code_parts.append('                        parameters.append(circuit_params)')
        code_parts.append('                    else:')
        code_parts.append('                        # Create mock parameters for this circuit')
        code_parts.append('                        mock_params = [MockParameter(f"param_{i}_{j}") for j in range(2)]')
        code_parts.append('                        # Add these parameters to the circuit so they can be found')
        code_parts.append('                        if hasattr(circuit, "_param_list"):')
        code_parts.append('                            circuit._param_list.extend(mock_params)')
        code_parts.append('                            circuit.parameters = MockParameterView(circuit._param_list)')
        code_parts.append('                        parameters.append(mock_params)')
        code_parts.append('            else:')
        code_parts.append('                # Parameters exist but ensure they are in the circuits')
        code_parts.append('                for i, (circuit, param_list) in enumerate(zip(circuits, parameters)):')
        code_parts.append('                    if hasattr(circuit, "_param_list") and param_list:')
        code_parts.append('                        # Add parameters to circuit if they are not already there')
        code_parts.append('                        for param in param_list:')
        code_parts.append('                            if param not in circuit._param_list:')
        code_parts.append('                                circuit._param_list.append(param)')
        code_parts.append('                        circuit.parameters = MockParameterView(circuit._param_list)')
        code_parts.append('')
        
        # Add the function body with proper indentation
        body_lines = func_info['body'].split('\n')


        transformed = []
        for line in body_lines:
            stripped = line.lstrip()
            indent = line[:len(line) - len(stripped)]

            # 1) Keep your offset override
            if stripped.startswith('offset ='):
                transformed.append(f"{indent}offset = np.identity(circuit.num_parameters)  # use full identity")

            # 2) Cast result slices to numpy arrays
            elif stripped.startswith('result = results.values'):
                transformed.append(line)
                transformed.append(f"{indent}result = np.array(result)  # ensure numeric array for subtraction")

            # 3) Resize before plus
            elif stripped.startswith('plus ='):
                transformed.append(f"{indent}parameter_values_ = np.resize(np.array(parameter_values_).flatten(), (circuit.num_parameters,))")
                transformed.append(line)

            # 4) Resize before minus
            elif stripped.startswith('minus ='):
                transformed.append(f"{indent}parameter_values_ = np.resize(np.array(parameter_values_).flatten(), (circuit.num_parameters,))")
                transformed.append(line)

            # 5) Everything else unchanged
            else:
                transformed.append(line)

        body_lines = transformed

        
    
        # now append them to your code_parts below…
        for line in body_lines:
            if line.strip():
                code_parts.append(f'    {line}')
            else:
                code_parts.append('')
        
        """
        # Add circuit detection with proper error handling
        for circuit in func_info['circuits']:
            if circuit.location == 'created':
                code_parts.append(f'    try:')
                code_parts.append(f'        print(f"Circuit {circuit.name}: {{type({circuit.name})}}")')
                code_parts.append(f'        if hasattr({circuit.name}, "num_qubits"):')
                code_parts.append(f'            print(f"Found QuantumCircuit: {circuit.name} with {{getattr({circuit.name}, \\"num_qubits\\", 0)}} qubits")')
                code_parts.append(f'            print(f"Parameters in {circuit.name}: {{len(getattr({circuit.name}, \\"parameters\\", []))}}")')
                code_parts.append(f'            if hasattr({circuit.name}, "parameters") and {circuit.name}.parameters:')
                code_parts.append(f'                print(f"Parameter names: {{[str(p) for p in {circuit.name}.parameters]}}")')
                code_parts.append(f'    except Exception as e:')
                code_parts.append(f'        print(f"Error analyzing circuit {circuit.name}: {{e}}")')
                code_parts.append('')
        """
        
        return '\n'.join(code_parts)
    
    def _modify_class_for_mocking(self, class_code: str, class_name: str) -> str:
        """Modify class definitions to work with mock parent classes."""
        lines = class_code.split('\n')
        modified_lines = []
        
        for line in lines:
            # Replace known parent classes with mock versions
            if 'BaseEstimatorGradient' in line and 'class' in line:
                line = line.replace('BaseEstimatorGradient', 'MockBaseEstimatorGradient')
            elif 'BaseEstimator' in line and 'class' in line and 'BaseEstimatorGradient' not in line:
                line = line.replace('BaseEstimator', 'MockBaseEstimator')
            
            # Handle super() calls in __init__ methods
            if 'super().__init__(' in line:
                # Add error handling around super() calls
                indent = len(line) - len(line.lstrip())
                indent_str = ' ' * indent
                modified_lines.append(f'{indent_str}try:')
                modified_lines.append(f'{indent_str}    {line.strip()}')
                modified_lines.append(f'{indent_str}except Exception as e:')
                modified_lines.append(f'{indent_str}    # Super call failed, initialize manually')
                modified_lines.append(f'{indent_str}    pass')
                continue
            
            modified_lines.append(line)
        
        return '\n'.join(modified_lines)
    
    def _generate_parameter_instance(self, param: ParameterInfo, containing_class: str = None) -> str:
        """Generate code to instantiate a parameter with intelligent constructor parameter detection."""
        if param.name == 'self':
            if containing_class:
                # Get constructor parameters for this class
                constructor_params = self.class_constructors.get(containing_class, [])
                
                # Generate minimal constructor call based on common patterns
                if constructor_params:
                    # Try to create a minimal instance by providing simple defaults
                    simple_args = []
                    for param_name in constructor_params[:5]:  # Limit to first 5 params
                        if 'estimator' in param_name.lower():
                            simple_args.append('MockBaseEstimator()')  # Use mock estimator
                        elif 'epsilon' in param_name.lower():
                            simple_args.append('0.01')  # Small positive float for epsilon
                        elif 'qubit' in param_name.lower():
                            simple_args.append('2')  # Default qubit count
                        elif 'state' in param_name.lower() or 'circuit' in param_name.lower():
                            simple_args.append('QuantumCircuit(2)')
                        elif 'oracle' in param_name.lower():
                            simple_args.append('QuantumCircuit(2)')
                        elif param_name.lower() == 'options':
                            simple_args.append('None')  # Options can be None
                        elif 'method' in param_name.lower():
                            simple_args.append('"central"')  # Default method
                        else:
                            simple_args.append('None')
                    
                    if simple_args:
                        args_str = ', '.join(simple_args)
                        return f'''# Create instance
try:
    self = {containing_class}({args_str})  # Instance with minimal params
except Exception as e:
    print(f"Failed to create {containing_class}: {{e}}")
    # Create a basic mock instead
    self = type('Mock{containing_class}', (), {{
        '__init__': lambda self, *args, **kwargs: None,
        '__getattr__': lambda self, name: lambda *args, **kwargs: None
    }})()'''
                    else:
                        return f'''# Create instance
try:
    self = {containing_class}()  # Instance with no params
except Exception as e:
    print(f"Failed to create {containing_class}: {{e}}")
    # Create a basic mock instead
    self = type('Mock{containing_class}', (), {{
        '__init__': lambda self, *args, **kwargs: None,
        '__getattr__': lambda self, name: lambda *args, **kwargs: None
    }})()'''
                else:
                    # No constructor parameters found, create a basic mock
                    return f'''self = type('Mock{containing_class}', (), {{
    '__init__': lambda self, *args, **kwargs: None,
    '__getattr__': lambda self, name: lambda *args, **kwargs: None,
    'qubits': [],
    'num_qubits': 2
}})()  # Mock instance'''
            else:
                return '# self parameter - no containing class found'
        
        if param.name == 'options' and param.type_annotation == "dict":
            # This is likely a **options parameter
            return f'{param.name} = {{}}  # Empty options dict for **{param.name}'
        
        if param.default_value is not None:
            return f'{param.name} = {repr(param.default_value)}  # Using default value'
        
        if not param.is_custom_type:
            # Handle standard Python types
            if 'bool' in param.type_annotation.lower():
                return f'{param.name} = False  # Default bool value'
            elif 'int' in param.type_annotation.lower():
                return f'{param.name} = 1  # Default int value'
            elif 'str' in param.type_annotation.lower():
                return f'{param.name} = "default"  # Default string value'
            elif 'float' in param.type_annotation.lower():
                return f'{param.name} = 1.0  # Default float value'
            else:
                return f'{param.name} = None  # Unknown standard type: {param.type_annotation}'
        else:
            # Handle custom types
            base_type = self._extract_base_type(param.type_annotation)
            return f'{param.name} = {self._generate_mock_instance(base_type, param.name)}'
    
    def _generate_mock_instance(self, class_name: str, param_name: str) -> str:
        """Generate a mock instance for a custom class."""
        # Clean up the class name to remove any malformed brackets or generics
        clean_class_name = self._clean_class_name(class_name)
        
        # Special handling for known Qiskit types
        if clean_class_name == 'QuantumCircuit':
            return 'QuantumCircuit(2)  # Mock QuantumCircuit with 2 qubits and 2 parameters'
        elif clean_class_name == 'EstimationProblem':
            return '''type('MockEstimationProblem', (), {
        'state_preparation': QuantumCircuit(2),
        'grover_operator': QuantumCircuit(2),
        'num_qubits': 2
    })()  # Mock EstimationProblem'''
        elif 'Sampler' in clean_class_name or 'BaseSampler' in clean_class_name:
            return '''type('MockSampler', (), {
        '__init__': lambda self, *args, **kwargs: None,
        'run': lambda self, *args, **kwargs: type('MockJob', (), {
            'result': lambda self: type('MockResult', (), {
                'quasi_dists': [{}]
            })()
        })()
    })()  # Mock Sampler'''
        elif 'Estimator' in clean_class_name or 'BaseEstimator' in clean_class_name:
            return 'MockBaseEstimator()  # Mock Estimator'
        elif clean_class_name == 'ClassicalRegister':
            return 'ClassicalRegister(2)  # Mock ClassicalRegister with 2 bits'
        elif 'Register' in clean_class_name:
            return 'ClassicalRegister(2)  # Mock Register with 2 bits'
        elif clean_class_name == 'Options':
            return 'MockOptions()  # Mock Options'
        elif clean_class_name in ['int', 'float', 'str', 'bool']:
            # Handle basic types that might be annotated
            defaults = {'int': '1', 'float': '1.0', 'str': '"default"', 'bool': 'False'}
            return f'{defaults.get(clean_class_name, "None")}  # Default {clean_class_name} value'
        # Handle sequence types specifically
        elif 'Sequence' in clean_class_name:
            # Extract the inner type if possible
            inner_type = self._extract_sequence_inner_type(class_name)
            if inner_type == 'QuantumCircuit':
                return '[QuantumCircuit(2), QuantumCircuit(2)]  # Mock sequence of QuantumCircuits (consistent 2 qubits each)'
            elif inner_type == 'BaseOperator':
                return '[MockSparsePauliOp(), MockSparsePauliOp()]  # Mock sequence of observables'
            elif inner_type == 'Parameter':
                return '[MockParameter("param1"), MockParameter("param2")]  # Mock sequence of parameters'
            elif 'float' in inner_type.lower():
                return '[[0.1, 0.2], [0.3, 0.4]]  # Mock sequence of parameter values (2 params each)'
            elif 'Sequence' in inner_type:
                # Handle nested sequences like Sequence[Sequence[Parameter]]
                nested_inner = self._extract_sequence_inner_type(inner_type)
                if 'Parameter' in nested_inner:
                    return '''[
        [MockParameter("theta_0"), MockParameter("phi_0")], 
        [MockParameter("theta_1"), MockParameter("phi_1")]
    ]  # Mock sequence of parameter sequences'''
                elif 'float' in nested_inner.lower():
                    return '[[0.1, 0.2], [0.3, 0.4]]  # Mock sequence of float sequences'
                else:
                    return '[[None], [None]]  # Mock nested sequence'
            else:
                return f'[MockParameter("default1"), MockParameter("default2")]  # Mock sequence for {class_name}'
        else:
            # Create a more robust generic mock
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '', clean_class_name)  # Remove special characters
            return f'''type('Mock{safe_name}', (), {{
        '__init__': lambda self, *args, **kwargs: None,
        '__str__': lambda self: 'Mock{safe_name}',
        '__repr__': lambda self: 'Mock{safe_name}()',
        '__getattr__': lambda self, name: lambda *args, **kwargs: None,
        '__iter__': lambda self: iter([]),  # Make it iterable
        'qubits': [],
        'num_qubits': 2,
        'name': '{safe_name}',
    }})()  # Enhanced mock {safe_name} instance'''
    
    def _clean_class_name(self, class_name: str) -> str:
        """Clean up class name by removing generic type parameters and fixing malformed brackets."""
        if not class_name:
            return class_name
        
        # Remove generic type parameters like Sequence[QuantumCircuit] -> Sequence
        if '[' in class_name:
            # Find the first bracket and take everything before it
            base_name = class_name.split('[')[0]
            return base_name.strip()
        
        return class_name.strip()
    
    def _extract_sequence_inner_type(self, class_name: str) -> str:
        """Extract the inner type from sequence annotations like Sequence[QuantumCircuit]."""
        if '[' in class_name and ']' in class_name:
            # Extract content between first [ and last ]
            start = class_name.find('[')
            end = class_name.rfind(']')
            if start != -1 and end != -1 and end > start:
                inner = class_name[start + 1:end].strip()
                # Handle nested generics by taking the first type
                if ',' in inner and not ('[' in inner and ']' in inner):
                    inner = inner.split(',')[0].strip()
                return inner
        
        return 'Unknown'
    
    def _extract_base_type(self, type_annotation: str) -> str:
        """Extract base type from complex annotations like 'QuantumCircuit | None' or 'Sequence[QuantumCircuit]'."""
        if not type_annotation:
            return type_annotation
        
        # Remove Optional, Union, | None patterns first
        base_type = type_annotation.replace(' | None', '').replace('Optional[', '').replace('Union[', '')
        
        # Count brackets to ensure we remove them properly
        bracket_count = base_type.count('[') - base_type.count(']')
        if bracket_count > 0:
            # We have unclosed brackets, close them
            base_type += ']' * bracket_count
        elif bracket_count < 0:
            # We have extra closing brackets, remove them
            base_type = base_type.replace(']', '', -bracket_count)
        
        base_type = base_type.strip()
        
        # Handle multiple types separated by commas, take the first non-None type
        if ',' in base_type and not ('[' in base_type and ']' in base_type):
            # Only split on commas if we're not inside brackets
            parts = [p.strip() for p in base_type.split(',')]
            for part in parts:
                if part.lower() != 'none':
                    base_type = part
                    break
        
        return base_type
    
    def _is_custom_type(self, type_annotation: str) -> bool:
        """Check if a type annotation represents a custom (non-standard) type."""
        if not type_annotation:
            return False
        
        standard_types = {
            'int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set',
            'List', 'Dict', 'Tuple', 'Set', 'Optional', 'Union', 'Any',
            'None', 'NoneType'
        }
        
        base_type = self._extract_base_type(type_annotation)
        return base_type not in standard_types and base_type != ''
    
    def _annotation_to_string(self, annotation) -> str:
        """Convert AST annotation to string."""
        if hasattr(ast, 'unparse'):
            return ast.unparse(annotation)
        else:
            # Fallback for older Python versions
            if isinstance(annotation, ast.Name):
                return annotation.id
            elif isinstance(annotation, ast.Attribute):
                return f"{self._annotation_to_string(annotation.value)}.{annotation.attr}"
            elif isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
                left = self._annotation_to_string(annotation.left)
                right = self._annotation_to_string(annotation.right)
                return f"{left} | {right}"
            else:
                return str(annotation)
    
    def _ast_to_value(self, node):
        """Convert AST node to Python value."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id == 'None':
                return None
            elif node.id == 'True':
                return True
            elif node.id == 'False':
                return False
        return None
    
    def _find_parameter_circuits(self, func_node: ast.FunctionDef) -> List[CircuitInfo]:
        """Find QuantumCircuit parameters in function signature."""
        circuits = []
        
        for arg in func_node.args.args:
            if self._is_quantum_circuit_annotation(arg.annotation):
                circuit_info = CircuitInfo(
                    name=arg.arg,
                    location='parameter',
                    function_name=func_node.name,
                    line_number=func_node.lineno
                )
                circuits.append(circuit_info)
        
        return circuits
    
    def _find_created_circuits(self, func_node: ast.FunctionDef) -> List[CircuitInfo]:
        """Find QuantumCircuit instances created within the function."""
        circuits = []
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Assign):
                # Check for QuantumCircuit() constructor calls
                if isinstance(node.value, ast.Call):
                    if self._is_quantum_circuit_call(node.value):
                        # Get variable names being assigned
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                assignment_line = self._get_line_content(node.lineno)
                                circuit_info = CircuitInfo(
                                    name=target.id,
                                    location='created',
                                    function_name=func_node.name,
                                    line_number=node.lineno,
                                    assignment_line=assignment_line
                                )
                                circuits.append(circuit_info)
        
        return circuits
    
    def _is_quantum_circuit_annotation(self, annotation) -> bool:
        """Check if an annotation indicates QuantumCircuit type."""
        if annotation is None:
            return False
        
        # Handle different annotation formats
        if isinstance(annotation, ast.Name):
            return annotation.id == 'QuantumCircuit'
        elif isinstance(annotation, ast.Attribute):
            # Handle qiskit.QuantumCircuit or similar
            return annotation.attr == 'QuantumCircuit'
        elif isinstance(annotation, ast.BinOp):
            # Handle Union types like QuantumCircuit | None
            if isinstance(annotation.op, ast.BitOr):
                return (self._is_quantum_circuit_annotation(annotation.left) or 
                        self._is_quantum_circuit_annotation(annotation.right))
        elif isinstance(annotation, ast.Subscript):
            # Handle Optional[QuantumCircuit] or similar
            return self._is_quantum_circuit_annotation(annotation.slice)
        
        return False
    
    def _is_quantum_circuit_call(self, call_node: ast.Call) -> bool:
        """Check if a call node is creating a QuantumCircuit."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id == 'QuantumCircuit'
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr == 'QuantumCircuit'
        return False
    
    def _get_return_annotation(self, func_node: ast.FunctionDef) -> str:
        """Get the return type annotation as string."""
        if func_node.returns:
            return self._annotation_to_string(func_node.returns)
        return None
    
    def _extract_function_source(self, func_node: ast.FunctionDef) -> str:
        """Extract the source code of a function."""
        return self._extract_node_source(func_node)
    
    def _extract_class_source(self, class_node: ast.ClassDef) -> str:
        """Extract the source code of a class."""
        return self._extract_node_source(class_node)
    
    def _get_indentation(self, line: str) -> int:
        """Get the indentation level of a line."""
        return len(line) - len(line.lstrip())
    
    def _get_line_content(self, line_number: int) -> str:
        """Get the content of a specific line."""
        if 1 <= line_number <= len(self.code_lines):
            return self.code_lines[line_number - 1].strip()
        return ""
    
    def _find_containing_class(self, func_node: ast.FunctionDef) -> str:
        """Find the class that contains this function."""
        # This is a simplified approach - in a real implementation you'd need to track the AST hierarchy
        # For now, we'll look for classes in our class_definitions that might contain this function
        for class_name, class_code in self.class_definitions.items():
            if func_node.name in class_code:
                return class_name
        return None
    
    def _find_node_end_line(self, node: ast.AST, start_line: int) -> int:
        """Find the end line of an AST node."""
        # For AST nodes that have end_lineno, use it directly
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno
        
        # Get the base indentation of the node
        base_indent = self._get_indentation(self.code_lines[start_line])
        
        # Look through the rest of the lines to find where this node ends
        i = start_line + 1
        while i < len(self.code_lines):
            line = self.code_lines[i]
            
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('#'):
                i += 1
                continue
                
            # Check indentation
            current_indent = self._get_indentation(line)
            
            # If we find a line with same or less indentation
            if current_indent <= base_indent:
                # Check if it's a decorator for the next method/function
                if line.strip().startswith('@'):
                    return i
                # Check if it's a method/function definition
                elif line.strip().startswith('def ') or line.strip().startswith('class '):
                    return i
                # Check if it's at the same level as the original function
                elif current_indent == base_indent:
                    return i
                
            i += 1
        
        # If we reach the end of file
        return len(self.code_lines)
    
    def _extract_node_source(self, node: ast.AST) -> str:
        """Extract the source code for any AST node."""
        start_line = node.lineno - 1
        
        # Try to get end line from the node
        if hasattr(node, 'end_lineno') and node.end_lineno:
            end_line = node.end_lineno
        else:
            # Fall back to finding end by analyzing the structure
            end_line = self._find_node_end_line(node, start_line)
        
        return '\n'.join(self.code_lines[start_line:end_line])


"""# Example usage
if __name__ == "__main__":
    # python -m test.StaticCircuit  

    # For a file
    file = os.path.abspath("C:/Users/rical/OneDrive/Desktop/QSmell_Tool/qsmell-tool/qiskit_algorithms/eigensolvers/vqd.py")

    generator = FunctionExecutionGenerator()
    
    # Option 1: Generate executable for a specific function (original functionality)
    # executable_code = generator.analyze_and_generate_executable(file, "construct_circuit")
    # with open("executable_circuit.py", "w") as f:
    #     f.write(executable_code)
    
    # Option 2: Generate executables for ALL functions with QuantumCircuits
    print("Analyzing file for functions with QuantumCircuits...")
    
    # First, find all functions with circuits
    functions_with_circuits = generator.find_all_functions_with_circuits(open(file, 'r').read(), file)
    print(f"Found {len(functions_with_circuits)} functions with QuantumCircuits:")
    for func_name in functions_with_circuits:
        print(f"  - {func_name}")
    
    # Generate executable files for all of them
    output_directory = "generated_executables"
    executables = generator.analyze_and_generate_all_executables(file, output_directory)
    
    print(f"\nGenerated {len(executables)} executable files in '{output_directory}/' directory")
    
    # Option 3: Generate executables but don't save to files (just get the code)
    # executables = generator.analyze_and_generate_all_executables(file)
    # for func_name, code in executables.items():
    #     print(f"\n=== Executable code for {func_name} ===")
    #     print(code[:200] + "..." if len(code) > 200 else code)"""