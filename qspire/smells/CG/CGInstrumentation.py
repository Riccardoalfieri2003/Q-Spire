# Instrumentation setup
import qiskit
from qiskit import QuantumCircuit
import numpy as np

# Get the detector from the module namespace
detector = globals().get('_custom_gate_detector')
original_code = globals().get('_original_code', '')
original_lines = globals().get('_original_lines', [])

if detector is None:
    # Fallback if detector not found
    class DummyDetector:
        def track_custom_gate_with_id(self, str, int, *args, **kwargs): pass
        def track_custom_gate(self, *args, **kwargs): pass
        def track_function_call(self, *args): pass
        def track_circuit_creation(self, *args): pass
    detector = DummyDetector()

# Monkey patch QuantumCircuit to detect custom gates
original_unitary = QuantumCircuit.unitary
original_hamiltonian = getattr(QuantumCircuit, 'hamiltonian', None)
original_singlequbitunitary = getattr(QuantumCircuit, 'singlequbitunitary', None)


def patched_unitary(self, call_id, matrix, qubits, label=None):
    #print("Sono qui")
    #print(f"Call_id: {call_id}")
    detector.track_custom_gate_with_id('unitary', call_id, matrix, qubits, label=label)
    return original_unitary(self, matrix, qubits, label)

def patched_hamiltonian(self, call_id, hamiltonian, qubits, time=None, label=None):
    """Patched hamiltonian method that captures call_id."""
    detector.track_custom_gate_with_id('hamiltonian', call_id, hamiltonian, qubits, time=time, label=label)
    return original_hamiltonian(self, hamiltonian, qubits, time, label)

def patched_singlequbitunitary(self, call_id, matrix, qubit, label=None):
    """Patched singlequbitunitary method that captures call_id."""
    detector.track_custom_gate_with_id('singlequbitunitary', call_id, matrix, qubit, label=label)
    return original_singlequbitunitary(self, matrix, qubit, label)

# Apply patches
QuantumCircuit.unitary = patched_unitary
if original_hamiltonian:
    QuantumCircuit.hamiltonian = patched_hamiltonian
if original_singlequbitunitary:
    QuantumCircuit.singlequbitunitary = patched_singlequbitunitary

# Track QuantumCircuit creation
original_init = QuantumCircuit.__init__
def patched_init(self, *args, **kwargs):
    result = original_init(self, *args, **kwargs)
    detector.track_circuit_creation(self)
    return result

QuantumCircuit.__init__ = patched_init