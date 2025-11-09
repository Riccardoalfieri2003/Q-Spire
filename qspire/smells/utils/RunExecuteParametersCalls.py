import ast
import re
import importlib.util
import sys
import os
from typing import Dict, List, Any, Tuple
from collections import defaultdict

class RunExecuteParametersCalls:
    """
    Dynamically tracks function calls by executing the code with instrumentation.
    This will count actual runtime calls, including those in loops.
    """
    
    def __init__(self):
        self.target_functions = {'run', 'execute', 'assign_parameters', 'bind_parameters'}
        self.call_info = defaultdict(list)
        self.debug = False
    
    def analyze_file(self, filepath: str, debug: bool = False) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """
        Analyze a Python file by executing it with instrumentation.
        
        Args:
            filepath: Path to the Python file to analyze
            debug: If True, print debugging information
            
        Returns:
            Tuple of (run_calls, execute_calls, assign_parameters_calls, bind_parameters_calls)
            where each is a list of dictionaries with structure:
            [{'circuit': 'qc', 'row': 10, 'column_start': 5, 'column_end': 8}, ...]
        """

        #debug=True

        # Reset state
        self.call_info = defaultdict(list)
        self.debug = debug
        
        # Read source code
        with open(filepath, 'r', encoding="utf-8") as f:
            source_code = f.read()
        
        if debug:
            print(f"Dynamically analyzing file: {filepath}")
        
        # Parse and instrument the code
        tree = ast.parse(source_code)
        instrumentor = FunctionCallInstrumentor(self)
        instrumented_tree = instrumentor.visit(tree)
        ast.fix_missing_locations(instrumented_tree)
        
        # Compile and execute the instrumented code
        try:
            compiled_code = compile(instrumented_tree, filepath, 'exec')
            
            # Create execution environment with our tracking functions
            # Include common imports that might be needed
            exec_globals = {
                '__builtins__': __builtins__,
                '__file__': filepath,
                '__name__': '__main__',
                '_track_function_call': self._track_function_call,
                '_mock_run': self._mock_run,
            }
            
            # Try to import common quantum libraries to avoid import errors
            try:
                import qiskit
                exec_globals['qiskit'] = qiskit
                from qiskit import QuantumCircuit
                exec_globals['QuantumCircuit'] = QuantumCircuit
                from qiskit_aer import AerSimulator
                exec_globals['AerSimulator'] = AerSimulator
            except ImportError:
                if debug:
                    print("Warning: Could not import qiskit libraries")
            
            # Add current directory to Python path for imports
            original_path = sys.path[:]
            file_dir = os.path.dirname(os.path.abspath(filepath))
            if file_dir not in sys.path:
                sys.path.insert(0, file_dir)
            
            try:
                # Execute the instrumented code
                exec(compiled_code, exec_globals)
            except SystemExit as e:
                # Catch sys.exit() calls - this is expected behavior in analyzed code
                if debug:
                    print(f"Code called sys.exit({e.code}), continuing with analysis")
                # Continue normally - we've already collected the function calls
            except Exception as e:
                # Catch any other exceptions from the analyzed code
                if debug:
                    print(f"Exception during code execution: {e}")
                    import traceback
                    traceback.print_exc()
                # Continue normally - we've already collected function calls up to this point
            finally:
                # Restore original path
                sys.path[:] = original_path
                
        except Exception as e:
            # This catches compilation errors or other critical issues
            if debug:
                print(f"Error during code compilation or setup: {e}")
                import traceback
                traceback.print_exc()
        
        # Return the results as separate lists
        run_calls = self.call_info['run']
        execute_calls = self.call_info['execute']
        assign_parameters_calls = self.call_info['assign_parameters']
        bind_parameters_calls = self.call_info['bind_parameters']
        
        if debug:
            print(f"Dynamic analysis results:")
            print(f"  run: {len(run_calls)} calls")
            print(f"  execute: {len(execute_calls)} calls")
            print(f"  assign_parameters: {len(assign_parameters_calls)} calls")
            print(f"  bind_parameters: {len(bind_parameters_calls)} calls")
        
        return run_calls, execute_calls, assign_parameters_calls, bind_parameters_calls
    
    def _track_function_call(self, func_name: str, row: int, col_start: int, col_end: int, 
                           original_obj, original_method_name, circuit_info, *args, **kwargs):
        """Track a function call and execute the original function."""
        if self.debug:
            print(f"Tracking call to {func_name} at row {row}, circuit: {circuit_info}")
        
        # Create the call record with circuit information
        call_record = {
            'circuit': circuit_info,
            'row': row,
            'column_start': col_start,
            'column_end': col_end
        }
        
        self.call_info[func_name].append(call_record)
        
        # Execute the original function
        try:
            # For run, use mock functions to prevent actual quantum hardware execution
            # Execute is allowed to run normally
            if func_name == 'run':
                if self.debug:
                    print(f"Using mock execution for {func_name} to prevent quantum hardware access")
                return MockJob()
            else:
                # Get the original method from the object
                original_method = getattr(original_obj, original_method_name)
                return original_method(*args, **kwargs)
        except Exception as e:
            if self.debug:
                print(f"Error executing {func_name}: {e}")
            # If the original function fails, we still want to track the call
            # For testing purposes, return a mock result
            if func_name in ['run', 'execute']:
                return MockJob()
            else:
                # For assign_parameters, bind_parameters, return the original object
                return original_obj
    
    def _mock_run(self, *args, **kwargs):
        """Mock run function that returns a MockJob without executing on quantum hardware."""
        return MockJob()


class MockJob:
    """Mock job object for testing purposes that prevents actual quantum execution."""
    def __init__(self):
        self._result = MockResult()
    
    def result(self):
        return self._result
    
    def get_counts(self):
        return {'00': 100, '11': 100}
    
    def status(self):
        return 'DONE'
    
    def job_id(self):
        return 'mock_job_12345'


class MockResult:
    """Mock result object for testing purposes."""
    def __init__(self):
        self._counts = {'00': 100, '11': 100}
    
    def get_counts(self, circuit=None):
        return self._counts
    
    def get_statevector(self, circuit=None):
        # Return a simple mock statevector
        import numpy as np
        return np.array([0.70710678, 0, 0, 0.70710678])
    
    def get_memory(self, circuit=None):
        return ['00', '11'] * 50  # Mock memory data


class FunctionCallInstrumentor(ast.NodeTransformer):
    """AST transformer to instrument function calls."""
    
    def __init__(self, tracker):
        self.tracker = tracker
        self.target_functions = tracker.target_functions
    
    def visit_Call(self, node):
        self.generic_visit(node)
        
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            
            if method_name in self.target_functions:
                # Get column end position
                col_end = getattr(node, 'end_col_offset', node.col_offset + len(method_name))
                
                # Try to determine the circuit name
                circuit_info = self._determine_circuit_info(node, method_name)
                
                # For run, override with mock functions to prevent quantum hardware execution
                # Execute is allowed to run normally
                if method_name == 'run':
                    # Create instrumented call that tracks and uses mock execution
                    new_node = ast.Call(
                        func=ast.Name(id='_track_function_call', ctx=ast.Load()),
                        args=[
                            ast.Constant(value=method_name),           # function name
                            ast.Constant(value=node.lineno),          # row
                            ast.Constant(value=node.col_offset),      # column start
                            ast.Constant(value=col_end),              # column end
                            node.func.value,                          # the object (e.g., backend, qc)
                            ast.Constant(value=method_name),          # method name to call
                            circuit_info,                             # circuit information
                        ] + node.args,                               # original arguments
                        keywords=node.keywords,                      # original keyword arguments
                    )
                else:
                    # For execute, assign_parameters and bind_parameters, use normal execution
                    new_node = ast.Call(
                        func=ast.Name(id='_track_function_call', ctx=ast.Load()),
                        args=[
                            ast.Constant(value=method_name),           # function name
                            ast.Constant(value=node.lineno),          # row
                            ast.Constant(value=node.col_offset),      # column start
                            ast.Constant(value=col_end),              # column end
                            node.func.value,                          # the object (e.g., backend, qc)
                            ast.Constant(value=method_name),          # method name to call
                            circuit_info,                             # circuit information
                        ] + node.args,                               # original arguments
                        keywords=node.keywords,                      # original keyword arguments
                    )
                
                return ast.copy_location(new_node, node)
        
        return node
    
    def _determine_circuit_info(self, node, method_name):
        """
        Determine circuit information from the function call.
        Returns an AST node that will evaluate to the circuit name or None.
        """
        caller_obj = node.func.value
        
        # Case 1: Direct method call on circuit (e.g., qc.assign_parameters(), qc.bind_parameters())
        if method_name in ['assign_parameters', 'bind_parameters']:
            if isinstance(caller_obj, ast.Name):
                return ast.Constant(value=caller_obj.id)
            else:
                return ast.Constant(value=None)
        
        # Case 2: Backend method calls (e.g., backend.run(qc), backend.execute(qc))
        elif method_name in ['run', 'execute']:
            if node.args and isinstance(node.args[0], ast.Name):
                # First argument is likely the circuit
                return ast.Constant(value=node.args[0].id)
            elif node.args:
                # Try to extract circuit name from more complex expressions
                return self._extract_circuit_name_from_arg(node.args[0])
            else:
                return ast.Constant(value=None)
        
        return ast.Constant(value=None)
    
    def _extract_circuit_name_from_arg(self, arg_node):
        """
        Try to extract circuit name from complex argument expressions.
        """
        if isinstance(arg_node, ast.Name):
            return ast.Constant(value=arg_node.id)
        elif isinstance(arg_node, ast.Attribute):
            # Handle cases like obj.circuit
            if isinstance(arg_node.value, ast.Name):
                return ast.Constant(value=f"{arg_node.value.id}.{arg_node.attr}")
            else:
                return ast.Constant(value=None)
        elif isinstance(arg_node, ast.Subscript):
            # Handle cases like circuits[0]
            if isinstance(arg_node.value, ast.Name):
                return ast.Constant(value=f"{arg_node.value.id}[...]")
            else:
                return ast.Constant(value=None)
        else:
            return ast.Constant(value=None)


def count_functions(filepath: str, debug: bool = False) -> Dict[str, List[List[int]]]:
    """
    Analyze a file to track function calls by executing the code.
    
    Args:
        filepath: Path to the Python file to analyze
        debug: If True, print debugging information
        
    Returns:
        Dictionary with structure:
        {
            'run': [{'circuit': 'qc', 'row': 10, 'column_start': 5, 'column_end': 8}, ...],
            'execute': [{'circuit': 'qc2', 'row': 15, 'column_start': 10, 'column_end': 17}, ...],
            'assign_parameters': [{'circuit': 'qc', 'row': 20, 'column_start': 5, 'column_end': 21}, ...],
            'bind_parameters': [{'circuit': None, 'row': 25, 'column_start': 8, 'column_end': 22}, ...]
        }
    """
    tracker = RunExecuteParametersCalls()
    return tracker.analyze_file(filepath, debug)