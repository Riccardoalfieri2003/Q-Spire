# smells/NC/instrumentation_code.py
"""
RUN_CALLS = {}
BIND_CALLS = {}
_RUN_CALL_ID = 0
_BIND_CALL_ID = 0

import builtins

# Save original functions
_original_run = None
_original_bind = None

def patch_backend_run(backend):
    global _original_run
    if _original_run is None:
        _original_run = backend.run

    def new_run(circuit, *args, **kwargs):
        global _RUN_CALL_ID
        _RUN_CALL_ID += 1
        RUN_CALLS[_RUN_CALL_ID] = {
            'circuit_name': getattr(circuit, 'name', 'unknown'),
            'note': 'run called'
        }
        return _original_run(circuit, *args, **kwargs)

    backend.run = new_run

def patch_circuit_assign_parameters(circuit):
    global _original_bind
    if _original_bind is None:
        _original_bind = circuit.assign_parameters

    def new_bind(params):
        global _BIND_CALL_ID
        _BIND_CALL_ID += 1
        BIND_CALLS[_BIND_CALL_ID] = {
            'circuit_name': getattr(circuit, 'name', 'unknown'),
            'note': 'assign_parameters called'
        }
        return _original_bind(params)

    circuit.assign_parameters = new_bind
"""

# smells/NC/instrumentation_code.py
"""
RUN_CALLS = []
BIND_CALLS = []
_CALL_ID = 0

# This global dict will collect info about calls during execution
_call_info = {
    'assign_parameters_calls': [],
    'run_calls': []
}

def patch_circuit_assign_parameters(self, *args, **kwargs):
    import inspect

    # Get the source line and column info of the caller (where assign_parameters was called)
    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)

    call_data = {
        'circuit_name': getattr(self, '_qc_name', None),  # We'll inject this name when creating circuits
        'row': info.lineno,
        'column_start': None,  # Python doesn't easily provide this; leave None or estimate if needed
        'column_end': None,
    }
    _call_info['assign_parameters_calls'].append(call_data)

    # Call the original assign_parameters method (saved on the function itself)
    return patch_circuit_assign_parameters._original(self, *args, **kwargs)

def patch_circuit_run(self, *args, **kwargs):
    import inspect

    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)

    call_data = {
        'circuit_name': getattr(args[0], '_qc_name', None) if args else None,  # usually first arg is circuit
        'row': info.lineno,
        'column_start': None,
        'column_end': None,
    }
    _call_info['run_calls'].append(call_data)

    # Call original run method (saved on the function itself)
    return patch_circuit_run._original(self, *args, **kwargs)
"""




"""
def log_and_call_run(caller_name, lineno, col_start, col_end, obj, *args, **kwargs):
    global _CALL_ID
    _CALL_ID += 1
    RUN_CALLS.append({
        'call_id': _CALL_ID,
        'caller': caller_name,
        'lineno': lineno,
        'col_start': col_start,
        'col_end': col_end,
    })
    return obj.run(*args, **kwargs)

def log_and_call_assign(caller_name, lineno, col_start, col_end, obj, *args, **kwargs):
    global _CALL_ID
    _CALL_ID += 1
    BIND_CALLS.append({
        'call_id': _CALL_ID,
        'caller': caller_name,
        'lineno': lineno,
        'col_start': col_start,
        'col_end': col_end,
    })
    return obj.assign_parameters(*args, **kwargs)
"""










# NCInstrumentation.py
"""
# Global dict to collect info about calls during execution
_call_info = {
    'assign_parameters_calls': [],
    'run_calls': []
}

def patch_circuit_assign_parameters(self, *args, **kwargs):
    import inspect
    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)

    call_data = {
        'circuit_name': getattr(self, '_qc_name', None),
        'row': info.lineno,
        'column_start': None,
        'column_end': None,
    }
    _call_info['assign_parameters_calls'].append(call_data)

    # Call original
    return patch_circuit_assign_parameters._original(self, *args, **kwargs)

def patch_circuit_run(self, *args, **kwargs):
    import inspect
    frame = inspect.currentframe().f_back
    info = inspect.getframeinfo(frame)

    call_data = {
        'circuit_name': getattr(args[0], '_qc_name', None) if args else None,
        'row': info.lineno,
        'column_start': None,
        'column_end': None,
    }
    _call_info['run_calls'].append(call_data)

    return patch_circuit_run._original(self, *args, **kwargs)

# NCInstrumentation.py (continue)

def patch_backend_run(backend):
    original_run = backend.run

    def wrapped_run(*args, **kwargs):
        # Log information here if you want, e.g. circuit name
        if args:
            circuit_or_circuits = args[0]
            # Try to get circuit name(s)
            name = getattr(circuit_or_circuits, 'name', None)
            _call_info['run_calls'].append({
                'circuit_name': name,
                'row': -1,
                'column_start': -1,
                'column_end': -1
            })
        return original_run(*args, **kwargs)

    backend.run = wrapped_run
    return backend

def t_assign_parameters(circuit):
    # Save original 'assign_parameters' and patch it
    patch_circuit_assign_parameters._original = circuit.assign_parameters
    circuit.assign_parameters = lambda *args, **kwargs: patch_circuit_assign_parameters(circuit, *args, **kwargs)
    return circuit

def log_and_call_run(caller_name, row, col_start, col_end, backend_obj, *args, **kwargs):
    # Collect info, then call original
    call_data = {
        'circuit_name': getattr(args[0], '_qc_name', None) if args else None,
        'row': row,
        'column_start': col_start,
        'column_end': col_end,
    }
    _call_info['run_calls'].append(call_data)
    return patch_circuit_run._original(backend_obj, *args, **kwargs)

def log_and_call_assign_parameters(caller_name, row, col_start, col_end, circuit_obj, *args, **kwargs):
    call_data = {
        'circuit_name': getattr(circuit_obj, '_qc_name', None),
        'row': row,
        'column_start': col_start,
        'column_end': col_end,
    }
    _call_info['assign_parameters_calls'].append(call_data)
    return patch_circuit_assign_parameters._original(circuit_obj, *args, **kwargs)
"""





from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

"""
_call_info = {'assign_parameters_calls': [], 'run_calls': []}

def get_call_info():
    # Return a copy to avoid accidental modification outside
    return {
        'assign_parameters_calls': _call_info['assign_parameters_calls'][:],
        'run_calls': _call_info['run_calls'][:]
    }


_original_assign_parameters = QuantumCircuit.assign_parameters
def patched_assign_parameters(self, *args, **kwargs):
    print(f"assign_parameters called on {getattr(self, '_qc_name', None)}")
    _call_info['assign_parameters_calls'].append({
        'circuit_name': getattr(self, '_qc_name', None),
        'row': None,
        'column_start': None,
        'column_end': None
    })
    return _original_assign_parameters(self, *args, **kwargs)

_original_run = AerSimulator.run
def patched_run(self, *args, **kwargs):
    print(f"run called on backend {self}")
    _call_info['run_calls'].append({
        'circuit_name': getattr(args[0], '_qc_name', None) if args else None,
        'row': None,
        'column_start': None,
        'column_end': None
    })
    return _original_run(self, *args, **kwargs)
"""



"""
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

_call_info = {'assign_parameters_calls': [], 'run_calls': []}

def get_call_info():
    # Return a copy to avoid accidental modification outside
    return {
        'assign_parameters_calls': _call_info['assign_parameters_calls'][:],
        'run_calls': _call_info['run_calls'][:]
    }

# Patch assign_parameters on QuantumCircuit instance
def t_assign_parameters(qc: QuantumCircuit):
    _original_assign_parameters = qc.assign_parameters

    def patched_assign_parameters(*args, **kwargs):
        print(f"[DEBUG] assign_parameters called on circuit: {getattr(qc, '_qc_name', None)}")
        _call_info['assign_parameters_calls'].append({
            'circuit_name': getattr(qc, '_qc_name', None),
            'row': None,
            'column_start': None,
            'column_end': None
        })
        return _original_assign_parameters(*args, **kwargs)

    qc.assign_parameters = patched_assign_parameters
    return qc
"""


"""
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

_call_info = {'assign_parameters_calls': [], 'run_calls': []}

def get_call_info():
    # Return a copy to avoid accidental modification outside
    return {
        'assign_parameters_calls': _call_info['assign_parameters_calls'][:],
        'run_calls': _call_info['run_calls'][:]
    }

# Patch assign_parameters on a QuantumCircuit instance
def t_assign_parameters(qc: QuantumCircuit):
    _original_assign_parameters = qc.assign_parameters

    def patched_assign_parameters(*args, **kwargs):
        print(f"[DEBUG] assign_parameters called on circuit: {getattr(qc, 'name', None)}")
        _call_info['assign_parameters_calls'].append({
            'circuit_name': getattr(qc, 'name', None),
            'row': None,
            'column_start': None,
            'column_end': None
        })
        return _original_assign_parameters(*args, **kwargs)

    qc.assign_parameters = patched_assign_parameters
    return qc

# Patch backend.run to log
def patch_backend_run(backend):
    original_run = backend.run

    def patched_run(*args, **kwargs):
        print(f"[DEBUG] backend.run called with args: {args}")
        _call_info['run_calls'].append({
            'args': str(args),
            'kwargs': str(kwargs)
        })
        return original_run(*args, **kwargs)

    backend.run = patched_run
    return backend
"""


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

# Helper to wrap backend.run
#def log_and_call_run(caller_name, row, column_start, column_end, backend, *args, **kwargs):
"""
def log_and_call_run(method_name, row, column_start, column_end, backend, circuit, circuit_varname, *args, **kwargs):
    circuit = args[0] if args else None
    circuit_name = getattr(circuit, 'name', None)
    print(f"[DEBUG] run called on circuit: {circuit_name} at row {row}, columns {column_start}-{column_end}")

    print("QC:", circuit)
    print("Type of QC:", type(circuit))

    _call_info['run_calls'].append({
        'circuit_name': circuit_varname,
        'circuit_internal_name': getattr(circuit, 'name', None),
        'row': row,
        'column_start': column_start,
        'column_end': column_end
    })
    return getattr(backend, method_name)(circuit, *args, **kwargs)
"""


"""
def log_and_call_run(method_name, row, column_start, column_end, backend, circuit, circuit_varname, *args, **kwargs):
    circuit_name = getattr(circuit, 'name', None)
    #print(f"[DEBUG] run called on circuit: {circuit_name} at row {row}, columns {column_start}-{column_end}")

    #print("QC:", circuit)
    #print("Type of QC:", type(circuit))

    _call_info['run_calls'].append({
        'circuit_name': circuit_varname,
        #'circuit_internal_name': circuit_name,
        'row': row,
        'column_start': column_start,
        'column_end': column_end
    })
    return getattr(backend, method_name)(circuit, *args, **kwargs)
"""


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



"""
def log_and_call_assign_parameters(
    caller_name,
    row,
    column_start,
    column_end,
    qc,
    circuit_varname,  # this should be a string like "circuit_1"
    parameters,       # this should be the dict or whatever parameters you're assigning
    *args,
    **kwargs
):
    # For debugging, print both the passed name and internal name
    print(f"[DEBUG] assign_parameters called on circuit var: {circuit_varname} (internal name: {getattr(qc, '_base_name', 'N/A')}) with parameters: {parameters} at row {row}, columns {column_start}-{column_end}")

    _call_info['assign_parameters_calls'].append({
        'circuit_name': circuit_varname,  # Use the passed variable name, same as in run
        'row': row,
        'column_start': column_start,
        'column_end': column_end
    })
    
    # Make sure to call the original method
    return getattr(qc, caller_name)(parameters, *args, **kwargs)
"""
"""
def log_and_call_assign_parameters(
    caller_name,
    row,
    column_start,
    column_end,
    qc,
    circuit_varname,
    parameters,       # This is now properly positioned
    *args,
    **kwargs
):
    #print(f"[DEBUG] assign_parameters called on circuit var: {circuit_varname} with parameters: {parameters} at row {row}, columns {column_start}-{column_end}")

    _call_info['assign_parameters_calls'].append({
        'circuit_name': circuit_varname,
        'row': row,
        'column_start': column_start,
        'column_end': column_end
    })
    
    # Handle case where parameters is None
    if parameters is None:
        return getattr(qc, caller_name)(*args, **kwargs)
    return getattr(qc, caller_name)(parameters, *args, **kwargs)"""




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


"""
def log_and_call_run(method_name, row, column_start, column_end, backend, circuits, circuit_varname, *args, **kwargs):
    # Handle both single circuits and lists of circuits
    circuit_names = []
    
    if isinstance(circuits, list):
        # For transpiled circuits, try to get names from original circuits
        if hasattr(circuits[0], '_original_circuit'):
            circuit_names = [getattr(c._original_circuit, 'name', None) for c in circuits]
        else:
            circuit_names = [getattr(c, 'name', None) for c in circuits]
    else:
        # Single circuit case
        circuit_names = [getattr(circuits, 'name', circuit_varname)]
    
    # Store the first circuit name if available (or use varname as fallback)
    detected_name = circuit_names[0] if circuit_names else circuit_varname
    
    _call_info['run_calls'].append({
        'circuit_name': detected_name,
        'row': row,
        'column_start': column_start,
        'column_end': column_end,
        'all_circuit_names': circuit_names  # Optional: track all names
    })
    
    return getattr(backend, method_name)(circuits, *args, **kwargs)
"""