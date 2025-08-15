from __future__ import annotations

# Add current directory and parent directory to Python path for local imports
import sys
import os

source_dir = r"C:\Users\rical\OneDrive\Desktop\QSmell_Tool\qsmell-tool\qiskit_algorithms\gradients\reverse"
parent_dir = r"C:\Users\rical\OneDrive\Desktop\QSmell_Tool\qsmell-tool\qiskit_algorithms\gradients"
grandparent_dir = r"C:\Users\rical\OneDrive\Desktop\QSmell_Tool\qsmell-tool\qiskit_algorithms"

sys.path.insert(0, source_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

# Handle imports with error handling
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Required imports
from collections.abc import Sequence
from qiskit.circuit import QuantumCircuit, Parameter, Gate, ParameterExpression
from qiskit.circuit.library import RXGate, RYGate, RZGate, CRXGate, CRYGate, CRZGate
import itertools
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import *
import numpy as np

# Enhanced mock classes with proper parameter handling
class MockParameter:
    def __init__(self, name="param"):
        self.name = name
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"MockParameter('{self.name}')"
    
    def __eq__(self, other):
        if isinstance(other, MockParameter):
            return self.name == other.name
        return False
    
    def __hash__(self):
        return hash(self.name)

class MockParameterVector:
    def __init__(self, name, length):
        self.name = name
        self.length = length
        self._params = [MockParameter(f"{name}[{i}]") for i in range(length)]
    
    def __getitem__(self, index):
        return self._params[index]
    
    def __iter__(self):
        return iter(self._params)
    
    def __len__(self):
        return self.length

class MockParameterView:
    """Mock for QuantumCircuit.parameters which returns a ParameterView-like object."""
    def __init__(self, params=None):
        self._params = params or []
        # Create a data attribute that contains the parameters
        self.data = self._params
    
    def __iter__(self):
        return iter(self._params)
    
    def __len__(self):
        return len(self._params)
    
    def __bool__(self):
        return len(self._params) > 0
    
    def __contains__(self, item):
        return item in self._params
    
    def index(self, item):
        # Try direct lookup first
        if item in self._params:
            return self._params.index(item)
        
        # If not found, try by name (for MockParameter objects)
        if hasattr(item, 'name'):
            for i, param in enumerate(self._params):
                if hasattr(param, 'name') and param.name == item.name:
                    return i
        
        # If still not found, try string representation
        item_str = str(item)
        for i, param in enumerate(self._params):
            if str(param) == item_str:
                return i
        
        # If really not found, raise ValueError like the real implementation
        raise ValueError(f"{item} is not in parameters list. Available parameters: {[str(p) for p in self._params]}")

class MockQuantumCircuit:
    """Enhanced mock QuantumCircuit with proper parameter handling - FIXED VERSION."""
    def __init__(self, num_qubits=2, num_clbits=0, name=None):
        self.num_qubits = num_qubits
        self.num_clbits = num_clbits
        self.name = name or "circuit"
        
        # Create mock qubits and clbits
        self.qubits = [f"qubit_{i}" for i in range(num_qubits)]
        self.clbits = [f"clbit_{i}" for i in range(num_clbits)]
        
        # Create mock parameters - start with some default parameters that match common usage
        self._param_list = [
            MockParameter(f"theta_{i}") for i in range(max(2, num_qubits))
        ]
        self.parameters = MockParameterView(self._param_list)
        
        # Other circuit attributes
        self.data = []  # Instructions
        self.global_phase = 0
    
    @property
    def num_parameters(self):
        """Return the number of parameters as an integer (CRITICAL FIX for numpy compatibility)."""
        # Ensure this returns an integer for numpy compatibility
        count = len(self._param_list)
        return int(count)  # Explicit conversion to int
    
    def add_parameter(self, param):
        """Add a parameter to the circuit."""
        if param not in self._param_list:
            self._param_list.append(param)
            self.parameters = MockParameterView(self._param_list)
        return param
    
    def assign_parameters(self, param_dict):
        """Mock parameter assignment."""
        new_circuit = MockQuantumCircuit(self.num_qubits, self.num_clbits, self.name)
        # Filter out assigned parameters
        remaining_params = [p for p in self._param_list if p not in param_dict]
        new_circuit._param_list = remaining_params
        new_circuit.parameters = MockParameterView(remaining_params)
        return new_circuit
    
    def bind_parameters(self, param_dict):
        """Alias for assign_parameters."""
        return self.assign_parameters(param_dict)
    
    def copy(self):
        """Create a copy of the circuit."""
        new_circuit = MockQuantumCircuit(self.num_qubits, self.num_clbits, self.name)
        new_circuit._param_list = self._param_list.copy()
        new_circuit.parameters = MockParameterView(new_circuit._param_list)
        return new_circuit
    
    def __getattr__(self, name):
        # Default method for any other attributes - return a lambda that returns self for chaining
        if name.startswith('__') and name.endswith('__'):
            # Don't mock dunder methods
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return lambda *args, **kwargs: self

# Override the standard QuantumCircuit with our mock
QuantumCircuit = MockQuantumCircuit

class MockBaseEstimator:
    def __init__(self, *args, **kwargs):
        pass
    
    def run(self, circuits, observables, parameter_values=None, **kwargs):
        # Return a mock result that has a 'result()' method
        class MockJob:
            def result(self):
                class MockResult:
                    def __init__(self):
                        # Create mock values based on the number of circuits
                        num_circuits = len(circuits) if hasattr(circuits, '__len__') else 1
                        self.values = [0.5 + 0.1 * i for i in range(num_circuits)]
                return MockResult()
        return MockJob()

class MockBaseEstimatorGradient:
    def __init__(self, estimator=None, options=None):
        self._estimator = estimator or MockBaseEstimator()
        self._options = options or {}
    
    def _get_local_options(self, options):
        return {**self._options, **options}

class MockOptions:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class MockSparsePauliOp:
    """Mock for SparsePauliOp observables."""
    def __init__(self, data=None, coeffs=None):
        self.data = data or ["II", "XX", "YY", "ZZ"]
        self.coeffs = coeffs or [1.0, 1.0, 1.0, 1.0]
        self.num_qubits = 2
    
    def __iter__(self):
        return iter(zip(self.data, self.coeffs))

# Mock the SparsePauliOp if it's used
try:
    from qiskit.quantum_info import SparsePauliOp
except ImportError:
    SparsePauliOp = MockSparsePauliOp

# Module-level functions
def gradient_lookup(gate: Gate) -> list[tuple[complex, QuantumCircuit]]:
    """Returns a circuit implementing the gradient of the input gate.

    Args:
        gate: The gate whose derivative is returned.

    Returns:
        The derivative of the input gate as list of ``(coeff, circuit)`` pairs,
        where the sum of all ``coeff * circuit`` elements describes the full derivative.
        The circuit is the unitary part of the derivative with a potential separate ``coeff``.
        The output is a list as derivatives of e.g. controlled gates can only be described
        as a sum of ``coeff * circuit`` pairs.

    Raises:
        NotImplementedError: If the derivative of ``gate`` is not implemented.
    """

    param = gate.params[0]
    if isinstance(gate, RXGate):
        derivative = QuantumCircuit(gate.num_qubits)
        derivative.rx(param, 0)
        derivative.x(0)
        return [(-0.5j, derivative)]
    if isinstance(gate, RYGate):
        derivative = QuantumCircuit(gate.num_qubits)
        derivative.ry(param, 0)
        derivative.y(0)
        return [(-0.5j, derivative)]
    if isinstance(gate, RZGate):
        derivative = QuantumCircuit(gate.num_qubits)
        derivative.rz(param, 0)
        derivative.z(0)
        return [(-0.5j, derivative)]
    if isinstance(gate, CRXGate):
        proj1 = QuantumCircuit(gate.num_qubits)
        proj1.rx(param, 1)
        proj1.x(1)

        proj2 = QuantumCircuit(gate.num_qubits)
        proj2.z(0)
        proj2.rx(param, 1)
        proj2.x(1)

        return [(-0.25j, proj1), (0.25j, proj2)]
    if isinstance(gate, CRYGate):
        proj1 = QuantumCircuit(gate.num_qubits)
        proj1.ry(param, 1)
        proj1.y(1)

        proj2 = QuantumCircuit(gate.num_qubits)
        proj2.z(0)
        proj2.ry(param, 1)
        proj2.y(1)

        return [(-0.25j, proj1), (0.25j, proj2)]
    if isinstance(gate, CRZGate):
        proj1 = QuantumCircuit(gate.num_qubits)
        proj1.rz(param, 1)
        proj1.z(1)

        proj2 = QuantumCircuit(gate.num_qubits)
        proj2.z(0)
        proj2.rz(param, 1)
        proj2.z(1)

        return [(-0.25j, proj1), (0.25j, proj2)]
    raise NotImplementedError("Cannot implement gradient for", gate)

def derive_circuit(
    circuit: QuantumCircuit, parameter: Parameter, check: bool = True
) -> Sequence[tuple[complex, QuantumCircuit]]:
    """Return the analytic gradient expression of the input circuit wrt. a single parameter.

    Returns a list of ``(coeff, gradient_circuit)`` tuples, where the derivative of the circuit is
    given by the sum of the gradient circuits multiplied by their coefficient.

    For example, the circuit::

           ┌───┐┌───────┐┌─────┐
        q: ┤ H ├┤ Rx(x) ├┤ Sdg ├
           └───┘└───────┘└─────┘

    returns the coefficient `-0.5j` and the circuit equivalent to::

           ┌───┐┌───────┐┌───┐┌─────┐
        q: ┤ H ├┤ Rx(x) ├┤ X ├┤ Sdg ├
           └───┘└───────┘└───┘└─────┘

    as the derivative of `Rx(x)` is `-0.5j Rx(x) X`.

    Args:
        circuit: The quantum circuit to derive.
        parameter: The parameter with respect to which we derive.
        check: If ``True`` (default) check that the parameter is valid and that no product
            rule is required.

    Returns:
        A list of ``(coeff, gradient_circuit)`` tuples.

    Raises:
        ValueError: If ``parameter`` is of the wrong type.
        ValueError: If ``parameter`` is not in this circuit.
        NotImplementedError: If a non-unique parameter is added, as the product rule is not yet
            supported in this function.
    """
    if check:
        # this is added as useful user-warning, since sometimes ``ParameterExpression``s are
        # passed around instead of ``Parameter``s
        if not isinstance(parameter, Parameter):
            raise ValueError(f"parameter must be of type Parameter, not {type(parameter)}.")

        if parameter not in circuit.parameters:
            raise ValueError(f"The parameter {parameter} is not in this circuit.")

        # check uniqueness
        seen_parameters: set[Parameter] = set()
        for instruction in circuit.data:
            # get parameters in the current operation
            new_parameters = set()
            for p in instruction.operation.params:
                if isinstance(p, ParameterExpression):
                    new_parameters.update(p.parameters)

            if duplicates := seen_parameters.intersection(new_parameters):
                raise NotImplementedError(
                    "Product rule is not supported, circuit parameters must be unique, but "
                    f"{duplicates} are duplicated."
                )

            seen_parameters.update(new_parameters)

    summands, op_context = [], []
    for i, op in enumerate(circuit.data):
        gate = op.operation
        op_context.append((op.qubits, op.clbits))
        if parameter in gate.params:
            coeffs_and_grads = gradient_lookup(gate)
            summands += [coeffs_and_grads]
        else:
            summands += [[(1, gate)]]

    gradient = []
    for product_rule_term in itertools.product(*summands):
        summand_circuit = QuantumCircuit(*circuit.qregs)
        c = complex(1)
        for i, term in enumerate(product_rule_term):
            c *= term[0]
            # Qiskit changed the format of the stored value. The newer Qiskit has this internal
            # method to go from the older (legacy) format to new. This logic may need updating
            # at some point if this internal method goes away.
            if hasattr(summand_circuit.data, "_resolve_legacy_value"):
                value = summand_circuit.data._resolve_legacy_value(term[1], *op_context[i])
                summand_circuit.data.append(value)
            else:
                summand_circuit.data.append([term[1], *op_context[i]])
        gradient += [(c, summand_circuit.copy())]

    return gradient

if __name__ == "__main__":

    print("Yoooo")

    circuit = QuantumCircuit(2)  # Mock QuantumCircuit with 2 qubits and 2 parameters
    parameter = type('MockParameter', (), {
            '__init__': lambda self, *args, **kwargs: None,
            '__str__': lambda self: 'MockParameter',
            '__repr__': lambda self: 'MockParameter()',
            '__getattr__': lambda self, name: lambda *args, **kwargs: None,
            '__iter__': lambda self: iter([]),  # Make it iterable
            'qubits': [],
            'num_qubits': 2,
            'name': 'Parameter',
        })()  # Enhanced mock Parameter instance
    check = True  # Using default value
    if check:
        # this is added as useful user-warning, since sometimes ``ParameterExpression``s are
        # passed around instead of ``Parameter``s
        if not isinstance(parameter, Parameter):
            raise ValueError(f"parameter must be of type Parameter, not {type(parameter)}.")

        if parameter not in circuit.parameters:
            raise ValueError(f"The parameter {parameter} is not in this circuit.")

        # check uniqueness
        seen_parameters: set[Parameter] = set()
        for instruction in circuit.data:
            # get parameters in the current operation
            new_parameters = set()
            for p in instruction.operation.params:
                if isinstance(p, ParameterExpression):
                    new_parameters.update(p.parameters)

            if duplicates := seen_parameters.intersection(new_parameters):
                raise NotImplementedError(
                    "Product rule is not supported, circuit parameters must be unique, but "
                    f"{duplicates} are duplicated."
                )

            seen_parameters.update(new_parameters)

    summands, op_context = [], []
    for i, op in enumerate(circuit.data):
        gate = op.operation
        op_context.append((op.qubits, op.clbits))
        if parameter in gate.params:
            coeffs_and_grads = gradient_lookup(gate)
            summands += [coeffs_and_grads]
        else:
            summands += [[(1, gate)]]

    gradient = []
    for product_rule_term in itertools.product(*summands):
        summand_circuit = QuantumCircuit(*circuit.qregs)
        c = complex(1)
        for i, term in enumerate(product_rule_term):
            c *= term[0]
            # Qiskit changed the format of the stored value. The newer Qiskit has this internal
            # method to go from the older (legacy) format to new. This logic may need updating
            # at some point if this internal method goes away.
            if hasattr(summand_circuit.data, "_resolve_legacy_value"):
                value = summand_circuit.data._resolve_legacy_value(term[1], *op_context[i])
                summand_circuit.data.append(value)
            else:
                summand_circuit.data.append([term[1], *op_context[i]])
        gradient += [(c, summand_circuit.copy())]

    sys.exit(0)  # return gradient