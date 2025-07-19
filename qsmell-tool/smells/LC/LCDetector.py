import builtins
import inspect
import math
import textwrap
import tempfile
import runpy
import os

from qiskit import QuantumCircuit
from smells.Detector import Detector

from smells.LC.LC import LC

def get_variable_name(obj, max_depth=5):
    """Try to find the variable name referring to `obj` from caller stack frames."""
    obj_id = id(obj)
    for frame_info in inspect.stack()[1:max_depth]:
        frame = frame_info.frame
        for var_name, var_val in frame.f_locals.items():
            if id(var_val) == obj_id:
                return var_name
        for var_name, var_val in frame.f_globals.items():
            if id(var_val) == obj_id:
                return var_name
    return None

# max parallel operations
def max_parallelism(circuit):
    """Calculate the maximum number of parallel operations in the circuit."""
    moments = []
    for instr, qargs, cargs in circuit.data:
        qindices = [q._index for q in qargs]
        placed = False
        for moment in moments:
            if not any(q in moment for q in qindices):
                moment.update(qindices)
                placed = True
                break
        if not placed:
            moments.append(set(qindices))
    return max(len(m) for m in moments) if moments else 0


def max_operations_per_qubit(circuit: QuantumCircuit) -> int:
    """Calculate the maximum number of operations on any single qubit in the circuit."""
    qubit_counts = [0] * circuit.num_qubits
    
    for instruction, qargs, _ in circuit.data:
        for qubit in qargs:
            qubit_index = qubit._index
            qubit_counts[qubit_index] += 1
    
    return max(qubit_counts)



from collections import defaultdict

def get_max_gate_errors(backend_properties) -> dict:
    """
    Returns a dictionary of maximum error rates for all active gates.
    Format: {'gate_name': max_error}
    Includes both quantum gates and measurement errors.
    """
    # Initialize dictionary to store errors for each gate type
    gate_errors = defaultdict(list)
    
    # Process all quantum gates
    for gate in backend_properties.gates:
        # Get the error parameter (assuming it's the first parameter)
        if gate.parameters and hasattr(gate.parameters[0], 'value'):
            gate_errors[gate.gate].append(gate.parameters[0].value)
    
    # Add measurement errors (readout errors)
    if backend_properties.qubits:
        readout_errors = []
        for qubit_props in backend_properties.qubits:
            # Find the readout error property (typically index 3)
            for prop in qubit_props:
                if getattr(prop, 'name', '') == 'readout_error':
                    readout_errors.append(prop.value)
                    break
        if readout_errors:
            gate_errors['measure'] = readout_errors
    
    # Calculate maximum error for each gate type
    return {gate: max(errors) for gate, errors in gate_errors.items() if errors}

def get_max_gate_error(backend_properties) -> dict:
    """
    Returns a dictionary with the single gate having the maximum error rate.
    Format: {'gate_name': max_error}
    """
    # Get all gate errors
    all_errors = get_max_gate_errors(backend_properties)  # Using our previous function
    
    if not all_errors:
        return {}
    
    # Find the gate with maximum error
    max_gate = max(all_errors.items(), key=lambda x: x[1])
    
    return {max_gate[0]: max_gate[1]}








@Detector.register(LC)
class LCDetector(Detector):

    def detect(self, source_code):
        # 1. Load instrumentation code from file
        with open("smells/LC/LCinstrumentation.py") as f:
            instrumentation_code = f.read()

        # 2. Combine instrumentation + source code + extraction logic
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(textwrap.dedent(instrumentation_code) + "\n")
            tmp.write(textwrap.dedent(source_code) + "\n")
            tmp.write("import builtins\n")
            tmp.write("builtins.__run_log_result__ = get_run_log()\n")  # Store result in builtins

        try:
            # 3. Run combined code
            runpy.run_path(tmp_path, run_name="__main__")

            # 4. Retrieve the run log from builtins
            run_log = getattr(builtins, "__run_log_result__", [])
            #return run_log  # Return the collected data
        finally:
            os.remove(tmp_path)
            if hasattr(builtins, "__run_log_result__"):
                del builtins.__run_log_result__


        # 4. Extract runtime circuit info
        smells = []

        for circuit, backend in run_log:

            max_gate_error=get_max_gate_error(backend.properties()) # this is the max error of the active gates of the backed
            (gate_name, error_value), = max_gate_error.items()  # Note the comma after the tuple!

            l = max_operations_per_qubit(circuit)
            c = max_parallelism(circuit)

            likelihood=math.pow(1-error_value,l*c)

            # Heuristic thresholds — adjust as needed
            if likelihood<0.5:
                
                circuit_name = get_variable_name(circuit) or circuit.name
                backend_class_name = backend.__class__.__name__ 

                smell = LC(
                    likelihood=likelihood,
                    error=max_gate_error,
                    l=l,
                    c=c,
                    backend=backend_class_name,
                    circuit_name=circuit_name,
                    explanation="",
                    suggestion=""
                )

                smells.append(smell)
            

        return smells
