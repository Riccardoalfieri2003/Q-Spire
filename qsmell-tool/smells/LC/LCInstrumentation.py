import builtins
import types
import inspect
from qiskit import QuantumCircuit

# === Internal storages ===
_run_log = []  # List of (QuantumCircuit, backend_instance)
_backend_instances = []
_already_patched = set()


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

def get_run_log():
    return _run_log