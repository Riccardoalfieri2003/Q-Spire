from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

_call_info = {
    'assign_parameters_calls': [],
    'run_calls': []
}

def get_call_info():
    # Return a copy to avoid accidental modification outside
    return {
        'assign_parameters_calls': _call_info['assign_parameters_calls'][:],
        'run_calls': _call_info['run_calls'][:]
    }


def log_and_call_run(method_name, row, column_start, column_end, backend, circuit, circuit_varname, *args, **kwargs):

    #print(circuit_varname)
    # Ensure circuit is actually the circuit object, not the name
    if isinstance(circuit, str):
        # This should never happen if instrumentation is correct
        raise ValueError("Circuit should be the object, not the name")
    
    circuit_name = getattr(circuit, 'name', None)
    #print(f"[DEBUG] run called on circuit: {circuit_varname} at row {row}, columns {column_start}-{column_end}")

    _call_info['run_calls'].append({
        'circuit_name': circuit_varname,
        'row': row,
        'column_start': column_start,
        'column_end': column_end
    })
    
    # Make sure we're passing the circuit object, not the name
    return getattr(backend, method_name)(circuit, *args, **kwargs)


def log_and_call_assign_parameters(
    caller_name,
    row,
    column_start,
    column_end,
    qc,
    circuit_varname,  # The variable name (e.g., "qc2")
    parameters,
    *args,
    **kwargs
):
    # Get the actual circuit name if variable name is missing
    circuit_name = circuit_varname or getattr(qc, 'name', str(qc))
    
    _call_info['assign_parameters_calls'].append({
        'circuit_name': circuit_name,
        'row': row,
        'parameters': parameters
    })
    
    # Original function call
    if parameters is None:
        return getattr(qc, caller_name)(*args, **kwargs)
    return getattr(qc, caller_name)(parameters, *args, **kwargs)
