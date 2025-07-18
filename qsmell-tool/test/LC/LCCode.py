"""import datetime
from qiskit.providers.fake_provider.generic_backend_v2 import GenericBackendV2
from qiskit.transpiler.coupling import CouplingMap
from qiskit.circuit.instruction import Instruction
from qiskit.providers

class MyFakeBackend(GenericBackendV2):
    def __init__(
        self,
        num_qubits: int,
        basis_gates: list[str] | None = None,
        *,
        coupling_map: list[list[int]] | CouplingMap | None = None,
        control_flow: bool = False,
        calibrate_instructions: bool | Instruction | None = None,
        noise_settings: dict,
        dtm: float | None = None,
        seed: int | None = None,
    ):

        self.noise_settings = noise_settings
        super().__init__(num_qubits, basis_gates,
            coupling_map=coupling_map,
            control_flow=control_flow,
            calibrate_instructions=calibrate_instructions,
            dtm=dtm,
            seed=seed)
        
    def _create_properties(self):
        now = datetime.datetime.utcnow()
        qubits = [
            [Nduv(now, 'T1', 50e3, 0, 'us'),
             Nduv(now, 'T2', 60e3, 0, 'us'),
             Nduv(now, 'frequency', 5.0e9, 0, 'GHz')],
            [Nduv(now, 'T1', 52e3, 0, 'us'),
             Nduv(now, 'T2', 61e3, 0, 'us'),
             Nduv(now, 'frequency', 5.1e9, 0, 'GHz')]
        ]

        gates = [
            Gate('u3', [0], [Nduv(now, 'gate_error', 0.001, 0, '')]),
            Gate('u3', [1], [Nduv(now, 'gate_error', 0.002, 0, '')]),
            Gate('cx', [0, 1], [Nduv(now, 'gate_error', 0.03, 0, '')])
        ]

        return BackendProperties(
            backend_name='my_fake_backend',
            backend_version='1.0.0',
            last_update_date=now,
            qubits=qubits,
            gates=gates,
            general=[]
        )

    def _get_noise_defaults(self, name: str, num_qubits: int) -> tuple:
        if name in self.noise_settings:
            return self.noise_settings[name]

        if num_qubits == 1: return None #_NOISE_DEFAULTS_FALLBACK["1-q"]
        return None #_NOISE_DEFAULTS_FALLBACK["multi-q"]
    



from qiskit.providers.fake_provider import GenericBackendV2

    
    
from qiskit_aer import Aer
from qiskit_aer.noise import NoiseModel
from qiskit import transpile, QuantumCircuit


# Load fake backend
#fake_backend = FakeManila()
cmap = CouplingMap.from_full(27)
noise_settings = {
    "cz": (7.992e-08, 8.99988e-07, 1e-5, 5e-3),
    "id": (2.997e-08, 5.994e-08, 9e-5, 1e-4),
    "rz": (0.0, 0.0),
    "sx": (2.997e-08, 5.994e-08, 9e-5, 1e-4),
    "x": (2.997e-08, 5.994e-08, 9e-5, 1e-4),
    "measure": (6.99966e-07, 1.500054e-06, 1e-5, 5e-3),
    "delay": (None, None),
    "reset": (None, None),
}

fake_backend = MyFakeBackend(num_qubits=27,
    basis_gates=['cz', 'id', 'rz', 'sx', 'x'],
    coupling_map=cmap,
    noise_settings=noise_settings)

print(fake_backend.properties())



# Create a noise model from fake backend
noise_model = NoiseModel.from_backend(fake_backend)

# Sample circuit
qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

# Transpile for the fake backend (uses same coupling map/basis gates)
qc_t = transpile(qc, fake_backend)

# Run with noise
simulator = Aer.get_backend('aer_simulator')
job = simulator.run(qc_t, noise_model=noise_model)
result = job.result()

# Output
#print("Counts with simulated noise:", result.get_counts())
"""


import datetime
from qiskit.providers.fake_provider.generic_backend_v2 import GenericBackendV2
from qiskit.transpiler.coupling import CouplingMap
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate
"""
class MyFakeBackend(GenericBackendV2):
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
    ):

        self.noise_settings = noise_settings
        super().__init__(num_qubits, basis_gates,
            coupling_map=coupling_map,
            control_flow=control_flow,
            dtm=dtm,
            seed=seed)

    def _get_noise_defaults(self, name: str, num_qubits: int) -> tuple:
        if name in self.noise_settings:
            return self.noise_settings[name]

        if num_qubits == 1:
            return None #_NOISE_DEFAULTS_FALLBACK["1-q"]
        return None #_NOISE_DEFAULTS_FALLBACK["multi-q"]
    
    def properties(self) -> BackendProperties:
        qubits = []
        gates = []

        # Create fake qubit T1, T2 and readout error info
        for _ in range(self.num_qubits):
            t1 = 100e3  # μs
            t2 = 100e3  # μs
            freq = 5.0  # GHz
            readout_err = 0.02
            qubits.append([
                {"date": datetime.utcnow(), "name": "T1", "unit": "μs", "value": t1},
                {"date": datetime.utcnow(), "name": "T2", "unit": "μs", "value": t2},
                {"date": datetime.utcnow(), "name": "frequency", "unit": "GHz", "value": freq},
                {"date": datetime.utcnow(), "name": "readout_error", "unit": "", "value": readout_err}
            ])

        for gate_name, err_tuple in self.noise_settings.items():
            if err_tuple[0] is not None:  # ignore delay/reset
                error = err_tuple[2] if len(err_tuple) > 2 else 0.0
                gates.append(Gate(
                    gate=gate_name,
                    qubits=[0],  # simplified: apply to qubit 0
                    parameters=[{"date": datetime.utcnow(), "name": "gate_error", "unit": "", "value": error}]
                ))

        return BackendProperties(
            backend_name=self.name,
            backend_version="1.0.0",
            last_update_date=datetime.utcnow(),
            qubits=qubits,
            gates=gates,
            general=[]
        )




# Load fake backend
#fake_backend = FakeManila()
cmap = CouplingMap.from_full(27)
noise_settings = {
    "cz": (7.992e-08, 8.99988e-07, 1e-5, 5e-3),
    "id": (2.997e-08, 5.994e-08, 9e-5, 1e-4),
    "rz": (0.0, 0.0),
    "sx": (2.997e-08, 5.994e-08, 9e-5, 1e-4),
    "x": (2.997e-08, 5.994e-08, 9e-5, 1e-4),
    "measure": (6.99966e-07, 1.500054e-06, 1e-5, 5e-3),
    "delay": (None, None),
    "reset": (None, None),
}

fake_backend = MyFakeBackend(
    num_qubits=27,
    basis_gates=['cz', 'id', 'rz', 'sx', 'x'],
    coupling_map=cmap,
    noise_settings=noise_settings
)

print(fake_backend)
"""


from qiskit.transpiler import CouplingMap, Target
#from qiskit.providers.backend import BackendV2 as GenericBackendV2
from qiskit.circuit.library import RZGate, XGate, SXGate, CZGate, IGate, Measure
from qiskit.circuit import QuantumCircuit
from datetime import datetime
from qiskit.transpiler import InstructionProperties


"""
class MyFakeBackend(GenericBackendV2):

    from qiskit.providers.options import Options

    @classmethod
    def _default_options(cls) -> Options:
        from qiskit.providers.options import Options
        return Options()

    @property
    def max_circuits(self) -> int:
        return 1000  # arbitrary limit for fake backend

    def run(self, circuits, **kwargs):
        raise NotImplementedError("This fake backend does not support execution.")
    
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
    ):
        super().__init__(
            num_qubits=num_qubits,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            control_flow=control_flow,
            dtm=dtm,
            seed=seed,
        )

        # Build and set the target
        self._target = self._build_target(num_qubits, noise_settings)

    def _build_target(self, num_qubits: int, noise_settings: dict) -> Target:
        target = Target()

        # Helper to simplify creation of InstructionProperties with error values
        def make_props(single=True, error=0.0):
            if single:
                return {(i,): InstructionProperties(error=error) for i in range(num_qubits)}
            else:
                return {
                    (i, j): InstructionProperties(error=error)
                    for i in range(num_qubits)
                    for j in range(num_qubits)
                    if i != j
                }

        # Map of gate name to class and qubit arity
        gate_map = {
            "rz": (RZGate(0.0), 1),        # Requires phi
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
            error = params[2] if len(params) > 2 else 0.0
            qubits = [(i,) for i in range(num_qubits)] if arity == 1 else [
                (i, j) for i in range(num_qubits) for j in range(num_qubits) if i != j
            ]
            props = make_props(single=(arity == 1), error=error)
            #target.add_instruction(gate, qubits=qubits, instruction_properties=props)
            target.add_instruction(gate, props)

        return target

    @property
    def target(self):
        return self._target
    

    @property
    def properties(self) -> dict:
        gate_errors = {}
        
        for gate_name in set(instr.name for instr, _ in self.target.instructions):
            # Get the first qubit pair and its error for this gate
            props_dict = self.target[gate_name]
            if props_dict:
                first_props = next(iter(props_dict.values()))  # Get first InstructionProperties
                if hasattr(first_props, "error"):
                    gate_errors[gate_name] = first_props.error
        
        return gate_errors



# Load fake backend
cmap = CouplingMap.from_full(27)
noise_settings = {
    "cz": (7.992e-08, 8.99988e-07, 1e-5, 5e-3),
    "id": (2.997e-08, 5.994e-08, 9e-5, 1e-4),
    "rz": (0.0, 0.0),
    "sx": (2.997e-08, 5.994e-08, 9e-5, 1e-4),
    "x": (2.997e-08, 5.994e-08, 9e-5, 1e-4),
    "measure": (6.99966e-07, 1.500054e-06, 1e-5, 5e-3),
    "delay": (None, None),
    "reset": (None, None),
}

fake_backend = MyFakeBackend(
    num_qubits=27,
    basis_gates=['cz', 'id', 'rz', 'sx', 'x'],
    coupling_map=cmap,
    noise_settings=noise_settings
)


errors = fake_backend.properties
print(errors)
"""




















"""

# For Qiskit 1.0+ (new location of fake backends)
#from qiskit_ibm_runtime.fake_provider import FakeWashington  # IBM-specific fakes
# OR for generic fakes:
#from qiskit.providers.fake_provider import Fake27QPulseV2  # Generic 27-qubit fake



#from qiskit_ibm_runtime.fake_provider.backends.washington import FakeWashingtonV2
#from qiskit.providers.fake_provider import GenericBackendV2

from qiskit_ibm_runtime import IBMBackend
def get_gate_errors_from_properties(backend):
    
    props = backend.properties()
    if props is None:
        return {}
    
    gate_errors = {}
    
    # Gate errors
    for gate in props.gates:
        gate_name = gate.gate
        for param in gate.parameters:
            if param.name == 'error':
                qubits = tuple(gate.qubits)
                gate_errors.setdefault(gate_name, {})[qubits] = param.value
    
    # Readout errors
    for qubit, qubit_props in enumerate(props.qubits):
        for param in qubit_props:
            if param.name == 'readout_error':
                gate_errors.setdefault('measure', {})[(qubit,)] = param.value
    
    return gate_errors


def get_gate_errors_from_target(backend):
    target = backend.target
    gate_errors = {}

    print(target)
    
    for instruction in target.instructions:
        if isinstance(instruction, tuple):
            instruction = instruction[0]
        
        gate_name = instruction.name
        props_dict = target[gate_name]
        
        for qubits, props in props_dict.items():
            if hasattr(props, 'error'):
                gate_errors.setdefault(gate_name, {})[qubits] = props.error
    
    return gate_errors


#print(get_gate_errors_from_target(IBMBackend))
print(IBMBackend.target.fget)  # Shows the property getter method
print(IBMBackend.target)  # Shows the docstring
"""







from datetime import datetime
from typing import Optional, Dict, List, Union
import numpy as np

#from qiskit.circuit.gate import Gate

from qiskit_ibm_runtime.models.backend_properties import Gate

from qiskit_ibm_runtime.models import (
    BackendStatus,
    BackendProperties,
    PulseDefaults,
    GateConfig,
    QasmBackendConfiguration,
    PulseBackendConfiguration,
    Nduv
)




"""
class MyFakeBackend(GenericBackendV2):

    from qiskit.providers.options import Options

    @classmethod
    def _default_options(cls) -> Options:
        from qiskit.providers.options import Options
        return Options()

    @property
    def max_circuits(self) -> int:
        return 1000  # arbitrary limit for fake backend

    def run(self, circuits, **kwargs):
        raise NotImplementedError("This fake backend does not support execution.")
    
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

    def properties(
        self, 
        refresh: bool = False, 
        datetime: Optional[datetime] = None
    ) -> Optional[BackendProperties]:
        if datetime is not None:
            raise NotImplementedError("Historical properties not supported for fake backend")
            
        if self._properties is None or refresh:
            self._properties = self._build_backend_properties()
            
        return self._properties

    def _build_backend_properties(self) -> BackendProperties:
        # Get current time for the properties timestamp
        now = datetime.now()
        
        # Build qubit properties (T1, T2, frequency, etc.)
        qubits = []
        for q in range(self._num_qubits):
            qubit_properties = [
                Nduv(name="T1", date=now, unit="µs", value=self._t1_values[q]),
                Nduv(name="T2", date=now, unit="µs", value=self._t2_values[q]),
                Nduv(name="frequency", date=now, unit="GHz", value=self._frequency_values[q]),
                Nduv(name="readout_error", date=now, unit="", value=self._get_readout_error(q)),
            ]
            qubits.append(qubit_properties)
        
        # Build gate properties
        gates = []
        for gate_name, params in self._noise_settings.items():
            if len(params) < 3 or params[2] is None:
                continue  # Skip gates with no error specified
                
            error = params[2]
            
            # Single-qubit gates
            if gate_name in ['id', 'rz', 'sx', 'x']:
                for q in range(self._num_qubits):
                    gate = Gate(
                        name=gate_name,
                        gate="{}".format(gate_name),
                        qubits=[q],
                        parameters=[
                            Nduv(name="gate_error", date=now, unit="", value=error),
                            Nduv(name="gate_length", date=now, unit="ns", value=params[0])
                        ]
                    )
                    gates.append(gate)
            
            # Two-qubit gates (like cz)
            elif gate_name == 'cz':
                for q1 in range(self._num_qubits):
                    for q2 in range(self._num_qubits):
                        if q1 != q2:
                            gate = Gate(
                                name=gate_name,
                                gate="{}".format(gate_name),
                                qubits=[q1, q2],
                                parameters=[
                                    Nduv(name="gate_error", date=now, unit="", value=error),
                                    Nduv(name="gate_length", date=now, unit="ns", value=params[0])
                                ]
                            )
                            gates.append(gate)
        
        # Build general properties
        properties = {
            "backend_name": self.name,
            "backend_version": "1.0.0",
            "last_update_date": now,
            "gates": gates,
            "qubits": qubits,
            "general": []
        }
        
        return BackendProperties.from_dict(properties)
    
    def _get_readout_error(self, qubit: int) -> float:
        if 'measure' in self._noise_settings:
            measure_params = self._noise_settings['measure']
            if len(measure_params) > 2:
                return measure_params[2]  # Return the error value
        return 0.01  # Default 1% readout error



# Create backend with custom noise
fake_backend = MyFakeBackend(
    num_qubits=5,
    noise_settings={
        'x': (35.5e-9, None, 1e-4),  # (duration, None, error_rate)
        'cz': (500e-9, None, 5e-3),
        'measure': (1000e-9, None, 2e-2)
    },
    t1_values=[75e-6, 80e-6, 90e-6, 100e-6, 110e-6],  # Custom T1s
    t2_values=[120e-6, 130e-6, 140e-6, 150e-6, 160e-6]  # Custom T2s
)

# Access properties just like a real backend
props = fake_backend.properties()
print("T1 times:", [q[0].value for q in props.qubits])  # Get all T1 values
print("X gate errors:", [g.parameters[0].value for g in props.gates if g.gate == 'x'])
"""




from qiskit.circuit.library import RZGate, XGate, SXGate, IGate, CZGate, Measure
from qiskit.transpiler import Target, InstructionProperties
from datetime import datetime
from typing import Optional, List, Dict
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import CouplingMap

class MyFakeBackend(GenericBackendV2):
    from qiskit.providers.options import Options

    @classmethod
    def _default_options(cls) -> Options:
        from qiskit.providers.options import Options
        return Options()

    @property
    def max_circuits(self) -> int:
        return 1000  # arbitrary limit for fake backend

    def run(self, circuits, **kwargs):
        raise NotImplementedError("This fake backend does not support execution.")
    
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

    """
    def _build_backend_properties(self) -> BackendProperties:
        now = datetime.now()
        qubits = []
        for q in range(self._num_qubits):
            qubit_properties = [
                Nduv(name="T1", date=now, unit="µs", value=self._t1_values[q]),
                Nduv(name="T2", date=now, unit="µs", value=self._t2_values[q]),
                Nduv(name="frequency", date=now, unit="GHz", value=self._frequency_values[q]),
                Nduv(name="readout_error", date=now, unit="", value=self._get_readout_error(q)),
            ]
            qubits.append(qubit_properties)
        
        gates = []
        for gate_name, params in self._noise_settings.items():
            if len(params) < 3 or params[2] is None:
                continue
                
            error = params[2]
            
            if gate_name in ['id', 'rz', 'sx', 'x']:
                for q in range(self._num_qubits):
                    gates.append(
                        Gate(
                            gate=gate_name,  # Changed from 'name' to 'gate'
                            name=gate_name,  # Keep both for compatibility
                            qubits=[q],
                            parameters=[
                                Nduv(name="gate_error", date=now, unit="", value=error),
                                Nduv(name="gate_length", date=now, unit="ns", value=params[0])
                            ]
                        )
                    )
            elif gate_name == 'cz':
                for q1 in range(self._num_qubits):
                    for q2 in range(self._num_qubits):
                        if q1 != q2:
                            gates.append(
                                Gate(
                                    gate=gate_name,  # Changed from 'name' to 'gate'
                                    name=gate_name,  # Keep both for compatibility
                                    qubits=[q1, q2],
                                    parameters=[
                                        Nduv(name="gate_error", date=now, unit="", value=error),
                                        Nduv(name="gate_length", date=now, unit="ns", value=params[0])
                                    ]
                                )
                            )
        
        return BackendProperties.from_dict({
            "backend_name": self.name,
            "backend_version": "1.0.0",
            "last_update_date": now,
            "gates": gates,
            "qubits": qubits,
            "general": []
        })
    """

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
props = fake_backend.properties()
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


max_errors = get_max_gate_error(props)
print(max_errors)