# NCDetector.py
import ast
from smells.NC.NCInstrumentor import NCInstrumentor
from smells.Detector import Detector
from smells.NC.NC import NC
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

from smells.OperationCircuitTracker import analyze_quantum_file

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

    def detect(self, file):

        circuits = analyze_quantum_file(file)

        for circuit in circuits:
            print(circuit)
            import pprint
            pprint.pp(circuits[circuit])
            print()
        
        return

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
