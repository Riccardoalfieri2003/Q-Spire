import ast
import contextlib
import time
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
                # qspire-fix: missing third-party modules are mocked instead of stopping the run
                from smells.utils import mock_missing
                ctx = mock_missing.active() if mock_missing.enabled() else contextlib.nullcontext()
                # qspire-fix: bound the in-process execution (checked at every function call)
                deadline = time.time() + float(os.environ.get("QSPIRE_NC_EXEC_TIMEOUT", "60"))

                def _deadline_tracer(frame, event, arg):
                    if time.time() > deadline:
                        raise TimeoutError("qspire: NC execution time limit")
                    return None

                previous_tracer = sys.gettrace()
                sys.settrace(_deadline_tracer)
                try:
                    with ctx:
                        if os.environ.get("QSPIRE_TRACKER", "") == "legacy":
                            exec(compiled_code, exec_globals)
                        else:
                            # qspire-fix: statement by statement, so that a failing statement
                            # does not stop the rest (this replaces AutoFix in static mode)
                            self._exec_tolerant(instrumented_tree, filepath, exec_globals)
                finally:
                    sys.settrace(previous_tracer)
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
    
    def _exec_tolerant(self, module, filename, g):
        """Run a module statement by statement (descending into the `__main__` block).

        A statement that raises is skipped. When it raises NameError, the undefined name is
        bound to a permissive mock and the statement is run again (as AutoFix would mock it);
        the calls recorded during the failed attempt are discarded first.
        """
        from smells.utils.mock_missing import Mock

        def is_main(node):
            t = getattr(node, "test", None)
            return (isinstance(node, ast.If) and isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                    and t.left.id == "__name__" and len(t.comparators) == 1
                    and isinstance(t.comparators[0], ast.Constant) and t.comparators[0].value == "__main__")

        def run_list(stmts):
            for node in stmts:
                if is_main(node):
                    run_list(node.body)
                    continue
                code = compile(ast.Module(body=[node], type_ignores=[]), filename, "exec")
                for _ in range(6):
                    saved = {k: len(v) for k, v in self.call_info.items()}
                    try:
                        exec(code, g)
                        break
                    except NameError as e:
                        m = re.search(r"name '(\w+)' is not defined", str(e))
                        if not m or m.group(1) in g:
                            break
                        for k in list(self.call_info):
                            del self.call_info[k][saved.get(k, 0):]
                        g[m.group(1)] = Mock(m.group(1))
                    except (SystemExit, TimeoutError):
                        raise
                    except Exception:
                        break

        run_list(module.body)

    def _track_function_call(self, func_name: str, row: int, column_start: int, column_end: int, 
                           original_obj, original_method_name, circuit_info, *args, **kwargs):
        """Track a function call and execute the original function."""
        if self.debug:
            print(f"Tracking call to {func_name} at row {row}, circuit: {circuit_info}")
        
        # Create the call record with circuit information
        call_record = {
            'circuit': circuit_info,
            'row': row,
            'column_start': column_start,
            'column_end': column_end
        }

        # qspire-fix: a pass manager's or a transpiler pass's run() transpiles circuits (or
        # DAGs) and does not execute them
        is_pass_manager = func_name == 'run' and ('PassManager' in type(original_obj).__name__
                                                  or _is_transpiler_pass(original_obj)
                                                  or (args and type(args[0]).__name__ in ('DAGCircuit', 'DAGDependency')))
        if is_pass_manager:
            try:
                return getattr(original_obj, original_method_name)(*args, **kwargs)
            except Exception:
                return args[0] if args else None

        self.call_info[func_name].append(call_record)

        # qspire-fix: primitive PUBs such as sampler.run([(circuit, parameter_values)]) bind
        # parameters at execution time, so they count as a binding as well
        if func_name == 'run' and args:
            pubs = args[0] if isinstance(args[0], (list, tuple)) else [args[0]]
            def _binds(p):
                if isinstance(p, tuple):  # (circuit, parameter_values, ...)
                    return len(p) >= 2 and p[1] is not None
                # coerced SamplerPub / EstimatorPub objects
                values = getattr(p, 'parameter_values', None)
                try:
                    return int(getattr(values, 'num_parameters', 0) or 0) > 0
                except Exception:
                    return False
            # qspire-fix: parameter values passed to run() next to the circuits, e.g.
            # estimator.run(circuits, observables, parameter_values) (V1 primitives and
            # gradients) or backend.run(circuits, parameter_binds=...) (Aer)
            extra = list(args[1:]) + [v for k, v in kwargs.items() if k in ('parameter_values', 'parameter_binds')]
            if any(_binds(p) for p in pubs) or any(_looks_like_values(x) for x in extra):
                self.call_info['assign_parameters'].append(dict(call_record))
        
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


def _is_transpiler_pass(obj):
    try:
        from qiskit.transpiler.basepasses import BasePass
        return isinstance(obj, BasePass)
    except Exception:
        return False


def _looks_like_values(x, depth=0):
    """True for parameter values: numbers, possibly nested in lists/tuples/arrays/dicts."""
    import numbers
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return x.size > 0 and np.issubdtype(x.dtype, np.number)
    except Exception:
        pass
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return False
    if isinstance(x, numbers.Number):
        return depth > 0
    if isinstance(x, dict):
        return len(x) > 0 and all(_looks_like_values(v, depth + 1) or isinstance(v, (list, tuple)) for v in x.values())
    if isinstance(x, (list, tuple)) and depth < 3:
        return len(x) > 0 and all(_looks_like_values(v, depth + 1) for v in x)
    return False


class FunctionCallInstrumentor(ast.NodeTransformer):
    """AST transformer to instrument function calls."""
    
    _PM_NAME = re.compile(r"(?i)(^|_)(pm|pass_?manager|passmanager)s?$")
    _PM_CALL = re.compile(r"(?i)pass_?manager")

    def __init__(self, tracker):
        self.tracker = tracker
        self.target_functions = tracker.target_functions
        self.pm_vars = set()

    def visit_Module(self, node):
        # qspire-fix: names bound to pass managers (PassManager(...), generate_preset_pass_manager(...));
        # their run() transpiles and is not an execution. Needed when the object itself is a mock.
        for n in ast.walk(node):
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
                f = n.value.func
                fname = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
                if self._PM_CALL.search(fname or ""):
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            self.pm_vars.add(t.id)
                        elif isinstance(t, ast.Attribute):
                            self.pm_vars.add(t.attr)
        self.generic_visit(node)
        return node

    _DAG_NAME = re.compile(r"(?i)dag")
    _PASS_NAME = re.compile(r"(?i)(^|_)pass(es)?_?$")

    def _runs_on_dag(self, node):
        # e.g. pass_.run(dag) or SomePass().run(circuit_to_dag(qc)): transpiler passes on DAGs
        if not node.args:
            return False
        a = node.args[0]
        if isinstance(a, ast.Name):
            return bool(self._DAG_NAME.search(a.id))
        if isinstance(a, ast.Attribute):
            return bool(self._DAG_NAME.search(a.attr))
        if isinstance(a, ast.Call):
            f = a.func
            fname = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            return bool(self._DAG_NAME.search(fname or ""))
        return False

    def _is_pass_manager(self, receiver):
        if isinstance(receiver, ast.Call):  # e.g. PassManager([...]).run(circuit)
            f = receiver.func
            fname = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            return bool(self._PM_CALL.search(fname or ""))
        name = receiver.id if isinstance(receiver, ast.Name) else receiver.attr if isinstance(receiver, ast.Attribute) else None
        return bool(name) and (name in self.pm_vars or bool(self._PM_NAME.search(name))
                               or bool(self._PASS_NAME.search(name)))

    def visit_Call(self, node):
        self.generic_visit(node)
        
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr

            if method_name == 'run' and (self._is_pass_manager(node.func.value) or self._runs_on_dag(node)):
                return node  # qspire-fix: pass manager or transpiler pass, not an execution

            if method_name in self.target_functions:
                # Get column end position
                column_end = getattr(node, 'end_col_offset', node.col_offset + len(method_name))
                
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
                            ast.Constant(value=column_end),              # column end
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
                            ast.Constant(value=column_end),              # column end
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