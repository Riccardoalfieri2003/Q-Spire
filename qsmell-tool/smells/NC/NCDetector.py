# NCDetector.py
import os
import ast
import importlib.util
from smells.NC.NCInstrumentor import NCInstrumentor
from smells.Detector import Detector
from smells.NC.NC import NC
"""
@Detector.register(NC)
class NCDetector(Detector):
    smell_cls = NC

    def __init__(self, smell_cls):
        super().__init__(smell_cls)

    def detect(self, code: str) -> list[NC]:
        # Import instrumentation module dynamically
        instr_path = os.path.join(os.path.dirname(__file__), 'NCInstrumentation.py')
        spec = importlib.util.spec_from_file_location("NCInstrumentation", instr_path)
        instrumentation_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(instrumentation_module)

        # Parse instrumentation code and user code
        instrumentation_tree = ast.parse(open(instr_path, encoding='utf-8').read())
        code_tree = ast.parse(code)

        # Instrument user code
        instrumentor = NCInstrumentor()
        instrumented_tree = instrumentor.visit(code_tree)
        ast.fix_missing_locations(instrumented_tree)

        # Combine trees
        combined_module = ast.Module(
            body=instrumentation_tree.body + instrumented_tree.body,
            type_ignores=[]
        )

        compiled = compile(combined_module, filename="<ast>", mode="exec")

        # Prepare environment
        import builtins
        env = {
            '__builtins__': builtins,
            'patch_circuit_assign_parameters': instrumentation_module.patch_circuit_assign_parameters,
            'patch_circuit_run': instrumentation_module.patch_circuit_run,
            '_call_info': instrumentation_module._call_info,
        }

        # Execute instrumented code
        exec(compiled, env)

        # Gather counts
        assign_calls = env['_call_info']['assign_parameters_calls']
        run_calls = env['_call_info']['run_calls']

        smells = []
        if len(run_calls) > len(assign_calls):
            smell = NC(
                run_calls=run_calls,
                bind_calls=assign_calls,
                explanation="",
                suggestion="",
            )
            smells.append(smell)

        return smells
"""


import ast
"""
@Detector.register(NC)
class NCDetector(Detector):

    smell_cls = NC

    def detect(self, source_code):
        # Parse AST just to locate static places (row/col)
        tree = ast.parse(source_code, type_comments=True)

        # Load collected runtime info
        call_info = get_call_info()

        run_calls = call_info['run_calls']
        assign_calls = call_info['assign_parameters_calls']

        smells = []

        print(run_calls)

        if len(run_calls) > len(assign_calls):
            smell = NC(
                run_calls=run_calls,
                bind_calls=assign_calls,
                run_count=len(run_calls),
                assign_parameters_count=len(assign_calls),
                explanation=f"Found {len(run_calls)} runs and {len(assign_calls)} assign_parameters",
                suggestion="Consider batching parameter bindings or reducing backend.run calls"
            )
            smells.append(smell)

        return smells
"""




"""
@Detector.register(NC)
class NCDetector(Detector):

    smell_cls = NC

    def detect(self, source_code):
        # You can parse AST if needed, e.g., for static analysis or code location.
        # If you don’t need it, you can skip parsing or just keep it minimal.
        # tree = ast.parse(source_code, type_comments=True)  # optional

        # Load runtime collected info about calls
        call_info = get_call_info()

        run_calls = call_info.get('run_calls', [])
        assign_calls = call_info.get('assign_parameters_calls', [])

        smells = []

        # Debug print — remove or replace with logger
        print(f"Run calls detected: {len(run_calls)}")
        print(f"Assign parameters calls detected: {len(assign_calls)}")

        # Example condition: more runs than assign_parameters suggests inefficiency
        if len(run_calls) > len(assign_calls):
            smell = NC(
                run_calls=run_calls,
                bind_calls=assign_calls,
                run_count=len(run_calls),
                assign_parameters_count=len(assign_calls),
                explanation=f"Found {len(run_calls)} runs and {len(assign_calls)} assign_parameters calls.",
                suggestion="Consider batching parameter bindings or reducing backend.run calls."
            )
            smells.append(smell)

        return smells
"""


import ast

"""
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

globals_dict = {
    'QuantumCircuit': QuantumCircuit,
    'AerSimulator': AerSimulator,
    # Also include your instrumentation functions and classes
    'patch_backend_run': patch_backend_run,
    't_assign_parameters': t_assign_parameters,
    #'log_and_call_run': log_and_call_run,
    #'log_and_call_assign_parameters': log_and_call_assign_parameters,
    'get_call_info': get_call_info,
}



@Detector.register(NC)
class NCDetector(Detector):

    smell_cls = NC

    def detect(self, source_code):
        # Parse original code to AST
        tree = ast.parse(source_code)

        # Apply instrumentation (this inserts the instrumentation code into the AST)
        instrumentor = NCInstrumentor()
        instrumented_tree = instrumentor.visit(tree)
        ast.fix_missing_locations(instrumented_tree)

        # Compile the instrumented AST back to code object
        instrumented_code = compile(instrumented_tree, filename="<instrumented>", mode="exec")


        # Run the instrumented code to collect runtime info
        #exec(compiled_code)
        exec(instrumented_code, globals_dict)

        # Now get the collected info from runtime
        #call_info = get_call_info()
        call_info = globals_dict['get_call_info']()

        print("[DEBUG] call_info:", call_info)

        # Then your existing detection logic
        run_calls = call_info.get('run_calls', [])
        assign_calls = call_info.get('assign_parameters_calls', [])

        smells = []

        if len(run_calls) > len(assign_calls):
            smell = NC(
                run_calls=run_calls,
                bind_calls=assign_calls,
                run_count=len(run_calls),
                assign_parameters_count=len(assign_calls),
                explanation=f"Found {len(run_calls)} runs and {len(assign_calls)} assign_parameters calls.",
                suggestion="Consider batching parameter bindings or reducing backend.run calls."
            )
            smells.append(smell)

        return smells
"""



import ast
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

from smells.NC.NCInstrumentation import get_call_info, log_and_call_assign_parameters, log_and_call_run

globals_dict = {
    'QuantumCircuit': QuantumCircuit,
    'AerSimulator': AerSimulator,
    #'patch_backend_run': patch_backend_run,
    #'t_assign_parameters': t_assign_parameters,
    'get_call_info': get_call_info,
    'log_and_call_run': log_and_call_run,
    'log_and_call_assign_parameters': log_and_call_assign_parameters,
}

@Detector.register(NC)
class NCDetector(Detector):

    smell_cls = NC

    def detect(self, source_code):
        # Parse code to AST
        tree = ast.parse(source_code)

        # Instrument the AST
        instrumentor = NCInstrumentor()
        instrumented_tree = instrumentor.visit(tree)
        ast.fix_missing_locations(instrumented_tree)

        # Compile and exec
        instrumented_code = compile(instrumented_tree, filename="<instrumented>", mode="exec")
        exec(instrumented_code, globals_dict)

        # Collect runtime info
        call_info = globals_dict['get_call_info']()

        #print("[DEBUG] call_info:", call_info)

        # Detect smells
        run_calls = call_info.get('run_calls', [])
        assign_calls = call_info.get('assign_parameters_calls', [])

        smells = []

        # Group calls by circuit name
        from collections import defaultdict

        run_counts = defaultdict(int)
        assign_counts = defaultdict(int)

        # Count run calls per circuit
        for call in run_calls:
            circuit_name = call['circuit_name']
            run_counts[circuit_name] += 1

        # Count assign_parameters calls per circuit
        for call in assign_calls:
            circuit_name = call['circuit_name']
            assign_counts[circuit_name] += 1

        # Check for each circuit if run calls > assign calls
        all_circuits = set(run_counts.keys()).union(set(assign_counts.keys()))
        for circuit in all_circuits:
            run_count = run_counts.get(circuit, 0)
            assign_count = assign_counts.get(circuit, 0)
            
            if run_count > assign_count:
                # Filter calls for this specific circuit
                circuit_run_calls = [c for c in run_calls if c['circuit_name'] == circuit]
                circuit_assign_calls = [c for c in assign_calls if c['circuit_name'] == circuit]
                
                smell = NC(
                    run_calls=circuit_run_calls,
                    assign_parameter_calls=circuit_assign_calls,
                    explanation="",
                    suggestion=""
                )
                smells.append(smell)

        return smells
