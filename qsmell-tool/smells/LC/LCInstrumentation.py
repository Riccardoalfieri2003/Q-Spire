import builtins
import types
import inspect
from qiskit import QuantumCircuit

# === Internal storages ===
_run_log = []  # List of (QuantumCircuit, backend_instance, circuit_name)
_backend_instances = []
_already_patched = set()

def _get_variable_name_for_object(obj, frame):
    """
    Try to find the variable name for an object by inspecting the calling frame.
    """
    try:
        # Get local and global variables from the calling frame
        frame_locals = frame.f_locals
        frame_globals = frame.f_globals
        
        # Search in locals first, then globals
        for var_name, var_value in frame_locals.items():
            if var_value is obj and not var_name.startswith('_'):
                return var_name
                
        for var_name, var_value in frame_globals.items():
            if var_value is obj and not var_name.startswith('_'):
                return var_name
                
        # If not found by direct reference, try to find by id (less reliable but worth trying)
        obj_id = id(obj)
        for var_name, var_value in frame_locals.items():
            if id(var_value) == obj_id and not var_name.startswith('_'):
                return var_name
                
    except Exception:
        pass
    
    return None


# === Patch .run method of any backend instance ===
def _patch_run_method(obj):
    if not hasattr(obj, 'run'):
        return

    if obj in _already_patched:
        return

    original_run = obj.run

    def run_wrapper(self, circuits, **kwargs):
        # Get the calling frame to inspect variable names
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find the frame where .run() was called
            caller_frame = frame.f_back
            
            # Track circuits
            if isinstance(circuits, QuantumCircuit):
                circuits_list = [circuits]
            else:
                circuits_list = list(circuits)

            for circ in circuits_list:
                # Try to get the variable name for this circuit
                circuit_name = _get_variable_name_for_object(circ, caller_frame)
                if not circuit_name:
                    circuit_name = getattr(circ, 'name', 'circuit')
                
                _run_log.append((circ, self, circuit_name))

        finally:
            del frame  # Prevent reference cycles
        
        # Track backend
        _backend_instances.append(self)
        return original_run(circuits, **kwargs)

    obj.run = types.MethodType(run_wrapper, obj)
    _already_patched.add(obj)

# === Patch builtins to monitor instance creation of backends ===
_original_init = builtins.__build_class__

def _custom_build_class(func, name, *bases, **kwargs):
    cls = _original_init(func, name, *bases, **kwargs)

    if inspect.isclass(cls) and hasattr(cls, 'run'):
        original_init = cls.__init__

        def custom_init(self, *a, **k):
            original_init(self, *a, **k)
            try:
                _patch_run_method(self)
            except Exception:
                pass

        cls.__init__ = custom_init
    return cls

builtins.__build_class__ = _custom_build_class

def get_run_log():
    """Get the current run log."""
    return _run_log.copy()  # Return a copy to avoid external modifications

def clear_run_log():
    """Clear the run log."""
    global _run_log, _backend_instances
    _run_log.clear()
    _backend_instances.clear()

def get_backend_instances():
    """Get backend instances."""
    return _backend_instances.copy()