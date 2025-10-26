import datetime
from qiskit.providers.fake_provider.generic_backend_v2 import GenericBackendV2
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler import CouplingMap, Target
from qiskit.circuit.library import *
from qiskit.transpiler import InstructionProperties

from typing import Optional, Dict, List, Union

from qiskit_ibm_runtime.models import (
    BackendStatus,
    BackendProperties,
    #PulseDefaults,
    GateConfig,
    QasmBackendConfiguration,
    #PulseBackendConfiguration,
    Nduv
)

from datetime import datetime

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
            # Single-qubit gates
            "x": (XGate(), 1),
            "y": (YGate(), 1),
            "z": (ZGate(), 1),
            "h": (HGate(), 1),
            "s": (SGate(), 1),
            "sdg": (SdgGate(), 1),
            "t": (TGate(), 1),
            "tdg": (TdgGate(), 1),
            "sx": (SXGate(), 1),
            "sxdg": (SXdgGate(), 1),
            "id": (IGate(), 1),
            "i": (IGate(), 1),
            "rz": (RZGate(0.0), 1),
            "rx": (RXGate(0.0), 1),
            "ry": (RYGate(0.0), 1),
            "p": (PhaseGate(0.0), 1),
            "u": (UGate(0.0, 0.0, 0.0), 1),
            "u1": (U1Gate(0.0), 1),
            "u2": (U2Gate(0.0, 0.0), 1),
            "u3": (U3Gate(0.0, 0.0, 0.0), 1),
            
            # Two-qubit gates
            "cx": (CXGate(), 2),
            "cnot": (CXGate(), 2),
            "cy": (CYGate(), 2),
            "cz": (CZGate(), 2),
            "ch": (CHGate(), 2),
            "crx": (CRXGate(0.0), 2),
            "cry": (CRYGate(0.0), 2),
            "crz": (CRZGate(0.0), 2),
            "cp": (CPhaseGate(0.0), 2),
            "cu": (CUGate(0.0, 0.0, 0.0, 0.0), 2),
            "cu1": (CU1Gate(0.0), 2),
            "cu3": (CU3Gate(0.0, 0.0, 0.0), 2),
            "swap": (SwapGate(), 2),
            "iswap": (iSwapGate(), 2),
            "dcx": (DCXGate(), 2),
            "ecr": (ECRGate(), 2),
            "rxx": (RXXGate(0.0), 2),
            "ryy": (RYYGate(0.0), 2),
            "rzz": (RZZGate(0.0), 2),
            "rzx": (RZXGate(0.0), 2),
            
            # Three-qubit gates
            "ccx": (CCXGate(), 3),
            "toffoli": (CCXGate(), 3),
            "ccz": (CCZGate(), 3),
            "cswap": (CSwapGate(), 3),
            "fredkin": (CSwapGate(), 3),
            
            # Special operations
            "measure": (Measure(), 1),
            "reset": (Reset(), 1),
            "barrier": (Barrier(1), 1),  # Arity will be adjusted
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
            if gate_name in ["x","y","z","h","s","sdg","t","tdg","sx","sxdg","id","i","rz","rx","ry","p","u","u1","u2","u3"]:
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
            elif gate_name in ["cx","cnot","cy","cz","ch","crx","cry","crz","cp","cu","cu1","cu3","swap","iswap","dcx","ecr","rxx","ryy","rzz","rzx"]:
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

            #Three-qubit gates
            elif gate_name in ["ccx","toffoli","ccz","cswap","fredkin"]:
                for q1 in range(self._num_qubits):
                    for q2 in range(self._num_qubits):
                        for q3 in range(self._num_qubits):
                            if q1 != q2 and q2!=q3:
                                gates.append({
                                    "gate": gate_name,
                                    "name": gate_name,
                                    "qubits": [q1, q2, q3],
                                    "parameters": [
                                        {"name": "gate_error", "date": now, "unit": "", "value": error},
                                        {"name": "gate_length", "date": now, "unit": "ns", "value": params[0]}
                                    ]
                                })
            
            #Special operations
            elif gate_name in ["measure","reset","barrier"]:
                gates.append({
                    "gate": gate_name,
                    "name": gate_name,
                    "qubits": [],
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
