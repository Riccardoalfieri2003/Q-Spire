import ast
import importlib.util
import tempfile
import os
import sys
from typing import Dict, List, Tuple, Any

from smells.CG.CG import CG
from smells.Detector import Detector

@Detector.register(CG)
class CGDetectorAlt:
    def __init__(self):
        self.custom_gates = []
        self.function_calls = {}
        self.circuits = []
        self.original_lines = []  # Store original code lines
        self.call_locations = {}  # Map call IDs to their original locations
        self.next_call_id = 0     # Counter for unique call IDs
        
    def detect_custom_gates(self, code_string: str) -> List[CG]:
        """
        Detects custom gates in quantum code using AST instrumentation.
        
        Args:
            code_string: The quantum code to analyze
            
        Returns:
            List of CG (Custom Gates) smell objects
        """
        # Store original code lines for reference
        self.original_lines = code_string.splitlines()
        
        # Parse the original code
        tree = ast.parse(code_string)
        
        # Create line mapping before instrumentation
        self._create_call_mapping(tree)
        
        # Add instrumentation
        instrumented_tree = self._instrument_code(tree)
        
        # Convert back to code
        instrumented_code = ast.unparse(instrumented_tree)

        # Execute instrumented code
        results = self._execute_instrumented_code(instrumented_code, code_string)
        
        # Convert detected custom gates to CG smell objects
        cg_smells = []
        for gate_info in self.custom_gates:
            
            cg_smell = CG(
                row=gate_info['line_number'],
                col=gate_info['column'],
                explanation="",
                suggestion=""
            )
            cg_smells.append(cg_smell)
        
        return cg_smells
    
    def _create_call_mapping(self, tree: ast.Module):
        """Create a mapping of function calls to their original locations."""
        self.call_locations = {}
        self.next_call_id = 0
        
        class CallMapper(ast.NodeVisitor):
            def __init__(self, detector):
                self.detector = detector
                
            def visit_Call(self, node):
                if isinstance(node.func, ast.Attribute):
                    # Check if this is a potential custom gate call
                    if node.func.attr in ['unitary', 'hamiltonian', 'singlequbitunitary']:
                        # Assign a unique ID to this call
                        call_id = self.detector.next_call_id
                        self.detector.next_call_id += 1
                        
                        # Store the original location info
                        self.detector.call_locations[call_id] = {
                            'line': node.lineno,
                            'col_offset': node.col_offset,
                            'end_line': getattr(node, 'end_lineno', node.lineno),
                            'end_col': getattr(node, 'end_col_offset', node.col_offset),
                            'function': node.func.attr
                        }
                        
                        # Add the call_id as a marker in the AST node
                        node.call_id = call_id
                        
                self.generic_visit(node)
        
        mapper = CallMapper(self)
        mapper.visit(tree)
    
    def _instrument_code(self, tree: ast.Module) -> ast.Module:
        """Add instrumentation to the AST while preserving call IDs."""
        transformer = CustomGateInstrumentationTransformer(self)
        return transformer.visit(tree)
    
    def _execute_instrumented_code(self, instrumented_code: str, original_code: str) -> Dict[str, Any]:
        """Execute the instrumented code safely."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(instrumented_code)
            temp_file = f.name
        
        try:
            spec = importlib.util.spec_from_file_location("quantum_analysis", temp_file)
            module = importlib.util.module_from_spec(spec)

            # Set up the module's namespace with our detector
            module.__dict__['_custom_gate_detector'] = self
            module.__dict__['_original_code'] = original_code
            module.__dict__['_original_lines'] = self.original_lines
            
            spec.loader.exec_module(module)

            # Get collected data from the module
            if hasattr(module, '_detector_results'):
                return module._detector_results
            return {}
            
        except Exception as e:
            print(f"Error executing instrumented code: {e}")
            return {}
        finally:
            os.unlink(temp_file)

class CustomGateInstrumentationTransformer(ast.NodeTransformer):
    """AST transformer that adds custom gate detection instrumentation."""
    
    def __init__(self, detector):
        self.detector = detector
    
    def visit_Module(self, node):
        """Add initialization code at the module level."""
        # Create instrumentation code as AST nodes
        instrumentation_code = self._create_instrumentation_ast()
        
        # Add setup code at the beginning
        node.body = instrumentation_code + node.body
        
        return self.generic_visit(node)
    
    def visit_Call(self, node):
        """Transform calls to custom gate functions to include call ID."""
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr in ['unitary', 'hamiltonian', 'singlequbitunitary'] and
            hasattr(node, 'call_id')):
            
            # Add the call_id as the first argument
            call_id_arg = ast.Constant(value=node.call_id)
            
            # Create a new call that includes the call_id
            new_args = [call_id_arg] + node.args
            new_node = ast.Call(
                func=node.func,
                args=new_args,
                keywords=node.keywords
            )
            
            # Copy location info
            new_node.lineno = node.lineno
            new_node.col_offset = node.col_offset
            if hasattr(node, 'end_lineno'):
                new_node.end_lineno = node.end_lineno
            if hasattr(node, 'end_col_offset'):
                new_node.end_col_offset = node.end_col_offset
            
            return new_node
        
        return self.generic_visit(node)
    
    def _create_instrumentation_ast(self):
        """Create instrumentation code as AST nodes."""
        
        with open("smells/CG/CGInstrumentation.py", "r", encoding="utf-8") as file:
            instrumentation_code = file.read()     

        return ast.parse(instrumentation_code).body

# Extension of the CGDetector class
@Detector.register(CG)
class CGDetectorAlt(CGDetectorAlt):
    
    def track_custom_gate_with_id(self, gate_type: str, call_id: str, *args, **kwargs):
        """Track when a custom gate is used, using the call ID for location info."""

        print(f"DEBUG: call_id={call_id}, gate_type={gate_type}, args={args}, type(args)={type(args)}")
        print(f"DEBUG: function_calls keys: {list(self.function_calls.keys())}")
        


        # Get the original location info using the call_id
        location_info = self.call_locations.get(call_id, {})
        line_number = location_info.get('line', -1)
        col_offset = location_info.get('col_offset', -1)
        end_line = location_info.get('end_line', line_number)
        end_col = location_info.get('end_col', col_offset)
        
        # Get the actual line content from original code
        line_content = ""
        if 1 <= line_number <= len(self.original_lines):
            line_content = self.original_lines[line_number - 1]
            
        # Extract just the function call part if possible
        if col_offset >= 0 and line_content:
            # Try to extract the relevant part of the line
            if col_offset < len(line_content):
                call_part = line_content[col_offset:]
                # Find the end of the function call (look for opening parenthesis)
                if '(' in call_part:
                    func_end = call_part.find('(')
                    func_name = call_part[:func_end + 1]
                    line_content = line_content[:col_offset] + func_name + "..."
            
        custom_gate_info = {
            'type': gate_type,
            #'args': str(args)[:100] + "..." if len(str(args)) > 100 else str(args),
            #'kwargs': kwargs,
            'line_number': line_number,
            'column': col_offset,
            'end_line': end_line,
            'end_column': end_col,
            'line_content': line_content.strip(),
            'call_id': call_id
        }
        self.custom_gates.append(custom_gate_info)
        
        # Also track in function calls
        if gate_type not in self.function_calls:
            self.function_calls[gate_type] = 0
        self.function_calls[gate_type] += 1
        
        # Debug print
        print(f"DEBUG: Custom gate detected - {gate_type} at line {line_number}, col {col_offset}")
        print(f"  Content: {line_content.strip()}")

        print()
    
    """
    def track_custom_gate(self, gate_type: str, *args, line_number=None, **kwargs):
        #Fallback method for tracking custom gates without call ID.
        if line_number is None:
            line_number = self._get_caller_line_number()
            
        # Get the actual line content from original code
        line_content = ""
        if 1 <= line_number <= len(self.original_lines):
            line_content = self.original_lines[line_number - 1].strip()
            
        custom_gate_info = {
            'type': gate_type,
            #'args': str(args)[:100] + "..." if len(str(args)) > 100 else str(args),
            #'kwargs': kwargs,
            'line_number': line_number,
            'column': -1,  # Unknown column
            'end_line': line_number,
            'end_column': -1,
            'line_content': line_content,
            'call_id': -1  # No call ID
        }
        self.custom_gates.append(custom_gate_info)
        
        # Also track in function calls
        if gate_type not in self.function_calls:
            self.function_calls[gate_type] = 0
        self.function_calls[gate_type] += 1
        
        # Debug print
        print(f"DEBUG: Custom gate detected - {gate_type} at line {line_number}")
    """
    
    """
    def track_function_call(self, func_name: str):
        #Track general function calls.
        if func_name not in self.function_calls:
            self.function_calls[func_name] = 0
        self.function_calls[func_name] += 1
    """
    
    def track_circuit_creation(self, circuit):
        #Track when a QuantumCircuit is created.
        self.circuits.append({
            'num_qubits': getattr(circuit, 'num_qubits', 0),
            'num_clbits': getattr(circuit, 'num_clbits', 0),
            'name': getattr(circuit, 'name', 'unnamed')
        })
    
    

    """
    def get_detection_summary(self) -> Dict[str, Any]:
        #Get a summary of the detection results (for backward compatibility).
        return {
            'custom_gates_count': len(self.custom_gates),
            'custom_gates_details': self.custom_gates,
            'total_function_calls': self.function_calls,
            'circuits_created': len(self.circuits),
            'smell_detected': len(self.custom_gates) > 0
        }
    """

    