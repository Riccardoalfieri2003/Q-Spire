# instrumentation.py
import builtins
from collections import Counter, defaultdict
import types
import inspect
from qiskit import QuantumCircuit

# === Internal storages ===
_circuits_created = []
_run_log = []  # List of (QuantumCircuit, backend_instance)
_backend_instances = []
_already_patched = set()

# === Patch QuantumCircuit to track creation ===
_original_qc_init = QuantumCircuit.__init__

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

def _qc_init_hook(self, *args, **kwargs):
    _original_qc_init(self, *args, **kwargs)
    self._user_created = True
    _circuits_created.append(self)

"""
def _qc_init_hook(self, *args, **kwargs):
    _original_qc_init(self, *args, **kwargs)
    _circuits_created.append(self)


import os

def _qc_init_hook(self, *args, **kwargs):
    _original_qc_init(self, *args, **kwargs)

    # Filter only user-defined circuits
    stack = inspect.stack()
    for frame_info in stack:
        filename = os.path.abspath(frame_info.filename)
        if (
            "site-packages" not in filename
            and "dist-packages" not in filename
            and "qiskit" not in filename.lower()
            and not filename.startswith("<frozen")
        ):
            _circuits_created.append(self)
            break
"""

def _qc_init_hook(self, *args, **kwargs):
    _original_qc_init(self, *args, **kwargs)

    # Mark this circuit as user-created
    self._user_created = False
    _circuits_created.append(self)


QuantumCircuit.__init__ = _qc_init_hook

# === Patch .run method of any backend instance ===
def _patch_run_method(obj):
    if not hasattr(obj, 'run'):
        return

    if obj in _already_patched:
        return

    original_run = obj.run

    def run_wrapper(self, circuits, **kwargs):
        # Track circuits
        if isinstance(circuits, QuantumCircuit):
            circuits_list = [circuits]
        else:
            circuits_list = list(circuits)

        for circ in circuits_list:
            _run_log.append((circ, self))

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

# === Public APIs ===
def get_all_circuits():
    return _circuits_created

def get_all_backends():
    return _backend_instances

def get_run_log():
    return _run_log

"""
def get_all_user_circuits():
    return [c for c in _circuits_created if "circuit" in c.name]
    return [c for c in _circuits_created if getattr(c, '_user_created', False)]
    #return [c for c in _circuits_created if "circuit" in c.name]
"""

def get_all_user_circuits():
    # Consider any circuit that was created in user code (via stack inspection) as user circuit
    return [c for c in _circuits_created if getattr(c, '_user_created', False)]

















# max parallel operations
def max_parallelism(circuit):
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
    """
    Calculate the maximum number of operations on any single qubit in the circuit.
    
    Args:
        circuit: QuantumCircuit to analyze
        
    Returns:
        int: Maximum operation count on any qubit
    """
    qubit_counts = [0] * circuit.num_qubits
    
    for instruction, qargs, _ in circuit.data:
        for qubit in qargs:
            qubit_index = qubit._index
            qubit_counts[qubit_index] += 1
    
    return max(qubit_counts)







from qiskit import QuantumCircuit
from test.LC.MyFakeBackend import MyFakeBackend, get_max_gate_error

backend = MyFakeBackend(
    num_qubits=5,
    noise_settings={
        'x': (35.5e-9, None, 1e-4),  # (duration, None, error_rate)
        'cz': (500e-9, None, 5e-3),
        'measure': (1000e-9, None, 2e-2)
    },
    t1_values=[75e-6, 80e-6, 90e-6, 100e-6, 110e-6],
    t2_values=[120e-6, 130e-6, 140e-6, 150e-6, 160e-6]
)

qc = QuantumCircuit(1)
qc.h(0)

backend.run(qc)


qc2=QuantumCircuit(3)
qc2.h(0)
qc2.x(2)
qc2.x(1)
backend.run(qc2)


"""
for circ in get_all_user_circuits():
    print(get_variable_name(circ) or circ.name)
    print(circ.name)
    print(circ)
"""



#Funzionano

#Prendi massimo errore per ogni backend
"""
for backend in get_all_backends():
    props=backend.properties()
    max_error=get_max_gate_error(props)
    print(f"{backend.name} has a max error of {max_error}")
"""









#prendo il nome do ogni circuito eseguito su ogni backend
for circ, backend in get_run_log():
    circuit_name = get_variable_name(circ) or circ.name
    backend_class_name = backend.__class__.__name__  # ← This is what you want
    print(f"Circuit {circuit_name} executed on {backend_class_name}")

    props=backend.properties()
    max_error=get_max_gate_error(props)
    print(f"{backend_class_name} has a max error of {max_error}")

    print(circ)

    max_op=max_operations_per_qubit(circ)
    max_par=max_parallelism(circ)
    print(f"l is {max_op}")
    print(f"c is {max_par}")

    print()
