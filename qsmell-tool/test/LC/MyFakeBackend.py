import datetime
from qiskit.providers.fake_provider.generic_backend_v2 import GenericBackendV2
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler import CouplingMap, Target
from qiskit.circuit.library import RZGate, XGate, SXGate, CZGate, IGate, Measure
from qiskit.transpiler import InstructionProperties

from typing import Optional, Dict, List, Union
import numpy as np

from qiskit_ibm_runtime.models import (
    BackendStatus,
    BackendProperties,
    PulseDefaults,
    GateConfig,
    QasmBackendConfiguration,
    PulseBackendConfiguration,
    Nduv
)

from qiskit.circuit.library import RZGate, XGate, SXGate, IGate, CZGate, Measure
from qiskit.transpiler import Target, InstructionProperties
from datetime import datetime
from typing import Optional, List, Dict
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap

class MyFakeBackend(GenericBackendV2):
    """
    Class that simulates a generic Backend

    Used to get the properties of the Backend without logging inside the account
    """
    from qiskit.providers.options import Options

    @classmethod
    def _default_options(cls) -> Options:
        from qiskit.providers.options import Options
        return Options()

    @property
    def max_circuits(self) -> int:
        return 1000  # arbitrary limit for fake backend

    def run(self, circuits, **kwargs):
        return None
        #raise NotImplementedError("This fake backend does not support execution.")
    
    def __init__(
        self,
        num_qubits: int,
        basis_gates: list[str] | None = None,
        *,
        coupling_map: list[list[int]] | CouplingMap | None = None,
        control_flow: bool = False,
        noise_settings: dict,
        dtm: float | None = None,
        seed: int | None = None,
        t1_values: Optional[List[float]] = None,
        t2_values: Optional[List[float]] = None,
        frequency_values: Optional[List[float]] = None,
    ):
        super().__init__(
            num_qubits=num_qubits,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            control_flow=control_flow,
            dtm=dtm,
            seed=seed,
        )
        
        self._noise_settings = noise_settings
        self._num_qubits = num_qubits
        self._t1_values = t1_values or [100e-6] * num_qubits  # Default 100μs
        self._t2_values = t2_values or [150e-6] * num_qubits  # Default 150μs
        self._frequency_values = frequency_values or [5.0e9] * num_qubits  # Default 5GHz
        self._target = self._build_target(num_qubits, noise_settings)
        self._properties = None  # Will be lazy-loaded

    def _build_target(self, num_qubits: int, noise_settings: dict) -> Target:
        """Build a Target object with instruction properties including errors."""
        target = Target()

        # Helper to create instruction properties
        def make_props(single=True, error=0.0, duration=0.0):
            if single:
                return {(i,): InstructionProperties(error=error, duration=duration) 
                       for i in range(num_qubits)}
            else:
                return {
                    (i, j): InstructionProperties(error=error, duration=duration)
                    for i in range(num_qubits)
                    for j in range(num_qubits)
                    if i != j
                }

        # Map of gate name to class and qubit arity
        gate_map = {
            "rz": (RZGate(0.0), 1),  # Requires phi
            "x": (XGate(), 1),
            "sx": (SXGate(), 1),
            "id": (IGate(), 1),
            "cz": (CZGate(), 2),
            "measure": (Measure(), 1)
        }

        for name, params in noise_settings.items():
            if name not in gate_map or params[0] is None:
                continue
                
            gate, arity = gate_map[name]
            duration = params[0]
            error = params[2] if len(params) > 2 else 0.0
            
            props = make_props(
                single=(arity == 1),
                error=error,
                duration=duration
            )
            target.add_instruction(gate, props)

        return target

    def properties(self, refresh: bool = False, datetime: Optional[datetime] = None) -> Optional[BackendProperties]:
        """Return backend properties matching IBM's BackendProperties format."""
        if datetime is not None:
            raise NotImplementedError("Historical properties not supported for fake backend")
            
        if self._properties is None or refresh:
            self._properties = self._build_backend_properties()
            
        return self._properties


    def _build_backend_properties(self) -> BackendProperties:
        """Construct BackendProperties object with gate errors and qubit properties."""
        now = datetime.now().isoformat()
        
        # Build qubit properties
        qubits = []
        for q in range(self._num_qubits):
            qubit_properties = [
                {"name": "T1", "date": now, "unit": "µs", "value": self._t1_values[q]},
                {"name": "T2", "date": now, "unit": "µs", "value": self._t2_values[q]},
                {"name": "frequency", "date": now, "unit": "GHz", "value": self._frequency_values[q]},
                {"name": "readout_error", "date": now, "unit": "", "value": self._get_readout_error(q)},
            ]
            qubits.append(qubit_properties)
        
        # Build gate properties
        gates = []
        for gate_name, params in self._noise_settings.items():
            if len(params) < 3 or params[2] is None:
                continue
                
            error = params[2]
            
            # Single-qubit gates
            if gate_name in ['id', 'rz', 'sx', 'x']:
                for q in range(self._num_qubits):
                    gates.append({
                        "gate": gate_name,
                        "name": gate_name,
                        "qubits": [q],
                        "parameters": [
                            {"name": "gate_error", "date": now, "unit": "", "value": error},
                            {"name": "gate_length", "date": now, "unit": "ns", "value": params[0]}
                        ]
                    })
            
            # Two-qubit gates
            elif gate_name == 'cz':
                for q1 in range(self._num_qubits):
                    for q2 in range(self._num_qubits):
                        if q1 != q2:
                            gates.append({
                                "gate": gate_name,
                                "name": gate_name,
                                "qubits": [q1, q2],
                                "parameters": [
                                    {"name": "gate_error", "date": now, "unit": "", "value": error},
                                    {"name": "gate_length", "date": now, "unit": "ns", "value": params[0]}
                                ]
                            })
        
        # Build the properties dictionary
        properties_dict = {
            "backend_name": "fake_backend",
            "backend_version": "1.0.0",
            "last_update_date": now,
            "gates": gates,
            "qubits": qubits,
            "general": []
        }
        
        return BackendProperties.from_dict(properties_dict)
        
    def _get_readout_error(self, qubit: int) -> float:
        """Get readout error for specified qubit."""
        if 'measure' in self._noise_settings:
            measure_params = self._noise_settings['measure']
            if len(measure_params) > 2:
                return measure_params[2]
        return 0.01  # Default 1% readout error


# Example usage
fake_backend = MyFakeBackend(
    num_qubits=5,
    noise_settings={
        'x': (35.5e-9, None, 1e-4),  # (duration, None, error_rate)
        'cz': (500e-9, None, 5e-3),
        'measure': (1000e-9, None, 2e-2)
    },
    t1_values=[75e-6, 80e-6, 90e-6, 100e-6, 110e-6],
    t2_values=[120e-6, 130e-6, 140e-6, 150e-6, 160e-6]
)

# Access properties
#props = fake_backend.properties()
"""
print("T1 times:", [q[0].value for q in props.qubits])
print("X gate errors:", [g.parameters[0].value for g in props.gates if g.gate == 'x'])
print("CZ gate errors:", [g.parameters[0].value for g in props.gates if g.gate == 'cz'])"""


"""
# Get maximum error for each gate type
gate_max_errors = {}

# Process single-qubit gates
single_qubit_gates = ['x', 'sx', 'rz', 'id']
for gate in single_qubit_gates:
    errors = [g.parameters[0].value for g in props.gates if g.gate == gate]
    if errors:
        gate_max_errors[gate] = max(errors)

# Process two-qubit gates (like CZ)
two_qubit_gates = ['cz']
for gate in two_qubit_gates:
    errors = [g.parameters[0].value for g in props.gates if g.gate == gate]
    if errors:
        gate_max_errors[gate] = max(errors)

# Process measurement
measure_errors = [q[3].value for q in props.qubits]  # index 3 is readout_error
if measure_errors:
    gate_max_errors['measure'] = max(measure_errors)

print("Maximum gate errors:")
for gate, error in gate_max_errors.items():
    print(f"{gate}: {error}")
"""


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


#max_errors = get_max_gate_error(props)
#print(max_errors)