from __future__ import annotations

# Add current directory and parent directory to Python path for local imports
import sys
import os

source_dir = r"C:\Users\rical\OneDrive\Desktop\QSmell_Tool\qsmell-tool\qiskit_algorithms\minimum_eigensolvers"
parent_dir = r"C:\Users\rical\OneDrive\Desktop\QSmell_Tool\qsmell-tool\qiskit_algorithms"
grandparent_dir = r"C:\Users\rical\OneDrive\Desktop\QSmell_Tool\qsmell-tool"

sys.path.insert(0, source_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

# Handle imports with error handling
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Required imports
from collections.abc import Callable, Sequence, Mapping, Iterable, MappingView
from dataclasses import dataclass
from qiskit.circuit import QuantumCircuit
from qiskit.primitives import BaseSampler, BaseEstimator, EstimatorResult
from qiskit.primitives.utils import init_observable, _circuit_key
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_algorithms.algorithm_job import AlgorithmJob
from typing import Any
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import *

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

# Class definitions (with mock parent classes)
class _DiagonalEstimatorResult(EstimatorResult):
    """A result from an expectation of a diagonal observable."""

    # TODO make each measurement a dataclass rather than a dict
    best_measurements: Sequence[Mapping[str, Any]] | None = None

class _DiagonalEstimator(MockBaseEstimator):
    """An estimator for diagonal observables."""

    def __init__(
        self,
        sampler: BaseSampler,
        aggregation: float | Callable[[Iterable[tuple[float, float]]], float] | None = None,
        callback: Callable[[Sequence[Mapping[str, Any]]], None] | None = None,
        **options,
    ) -> None:
        r"""Evaluate the expectation of quantum state with respect to a diagonal operator.

        Args:
            sampler: The sampler used to evaluate the circuits.
            aggregation: The aggregation function to aggregate the measurement outcomes. If a float
                this specified the CVaR :math:`\alpha` parameter.
            callback: A callback which is given the best measurements of all circuits in each
                evaluation.
            run_options: Options for the sampler.

        """
        try:
            super().__init__(options=options)
        except Exception as e:
            # Super call failed, initialize manually
            pass
        self._circuits: list[QuantumCircuit] = []  # See Qiskit pull request 11051
        self._parameters: list[MappingView] = []
        self._observables: list[SparsePauliOp] = []

        self.sampler = sampler
        if not callable(aggregation):
            aggregation = _get_cvar_aggregation(aggregation)

        self.aggregation = aggregation
        self.callback = callback
        self._circuit_ids: dict[int, QuantumCircuit] = {}
        self._observable_ids: dict[int, BaseOperator] = {}

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> AlgorithmJob:
        circuit_indices = []
        for circuit in circuits:
            key = _circuit_key(circuit)
            index = self._circuit_ids.get(key)
            if index is not None:
                circuit_indices.append(index)
            else:
                circuit_indices.append(len(self._circuits))
                self._circuit_ids[key] = len(self._circuits)
                self._circuits.append(circuit)
                self._parameters.append(circuit.parameters)
        observable_indices = []
        for observable in observables:
            index = self._observable_ids.get(id(observable))
            if index is not None:
                observable_indices.append(index)
            else:
                observable_indices.append(len(self._observables))
                self._observable_ids[id(observable)] = len(self._observables)
                converted_observable = init_observable(observable)
                _check_observable_is_diagonal(converted_observable)  # check it's diagonal
                self._observables.append(converted_observable)
        job = AlgorithmJob(
            self._call, circuit_indices, observable_indices, parameter_values, **run_options
        )
        job.submit()
        return job

    def _call(
        self,
        circuits: Sequence[int],
        observables: Sequence[int],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> _DiagonalEstimatorResult:
        job = self.sampler.run(
            [self._circuits[i] for i in circuits],
            parameter_values,
            **run_options,
        )
        sampler_result = job.result()
        samples = sampler_result.quasi_dists

        # a list of dictionaries containing: {state: (measurement probability, value)}
        evaluations: list[dict[int, tuple[float, float]]] = [
            {
                state: (probability, _evaluate_sparsepauli(state, self._observables[i]))
                for state, probability in sampled.items()
            }
            for i, sampled in zip(observables, samples)
        ]

        results = np.array([self.aggregation(evaluated.values()) for evaluated in evaluations])

        # get the best measurements
        best_measurements = []
        num_qubits = self._circuits[0].num_qubits
        for evaluated in evaluations:
            best_result = min(evaluated.items(), key=lambda x: x[1][1])
            best_measurements.append(
                {
                    "state": best_result[0],
                    "bitstring": bin(best_result[0])[2:].zfill(num_qubits),
                    "value": best_result[1][1],
                    "probability": best_result[1][0],
                }
            )

        if self.callback is not None:
            self.callback(best_measurements)

        return _DiagonalEstimatorResult(
            values=results, metadata=sampler_result.metadata, best_measurements=best_measurements
        )

# Module-level functions
def _get_cvar_aggregation(alpha: float | None) -> Callable[[Iterable[tuple[float, float]]], float]:
    """Get the aggregation function for CVaR with confidence level ``alpha``."""
    if alpha is None:
        alpha = 1
    elif not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be in [0, 1] but was {alpha}")

    # if alpha is close to 1 we can avoid the sorting
    if np.isclose(alpha, 1):

        def aggregate(measurements: Iterable[tuple[float, float]]) -> float:
            return sum(probability * value for probability, value in measurements)

    else:

        def aggregate(measurements: Iterable[tuple[float, float]]) -> float:
            # sort by values
            sorted_measurements = sorted(measurements, key=lambda x: x[1])

            accumulated_percent = 0.0  # once alpha is reached, stop
            cvar = 0.0
            for probability, value in sorted_measurements:
                cvar += value * min(probability, alpha - accumulated_percent)
                accumulated_percent += probability
                if accumulated_percent >= alpha:
                    break

            return cvar / alpha

    return aggregate

def _evaluate_sparsepauli(state: int, observable: SparsePauliOp) -> float:
    packed_uint8 = np.packbits(observable.paulis.z, axis=1, bitorder="little")
    state_bytes = np.frombuffer(state.to_bytes(packed_uint8.shape[1], "little"), dtype=np.uint8)
    reduced = np.bitwise_xor.reduce(packed_uint8 & state_bytes, axis=1)
    return np.sum(observable.coeffs * _PARITY[reduced])

def _check_observable_is_diagonal(observable: SparsePauliOp) -> None:
    is_diagonal = not np.any(observable.paulis.x)
    if not is_diagonal:
        raise ValueError("The observable must be diagonal.")

# Parameter instantiation
if __name__ == "__main__":
    # Create instance
    try:
        self = _DiagonalEstimator(None, None, None)  # Instance with minimal params
    except Exception as e:
        print(f"Failed to create _DiagonalEstimator: {e}")
        # Create a basic mock instead
        self = type('Mock_DiagonalEstimator', (), {
            '__init__': lambda self, *args, **kwargs: None,
            '__getattr__': lambda self, name: lambda *args, **kwargs: None
        })()
    circuits = [QuantumCircuit(2), QuantumCircuit(2)]  # Mock sequence of QuantumCircuits (consistent 2 qubits each)
    observables = [MockSparsePauliOp(), MockSparsePauliOp()]  # Mock sequence of observables
    parameter_values = [[0.1, 0.2], [0.3, 0.4]]  # Mock sequence of parameter values (2 params each)
    run_options = None  # Unknown standard type: dict

# Fix parameter consistency for gradient calculations
# Ensure parameter_values match circuit parameters exactly
    if "parameter_values" in locals() and "circuits" in locals():
        # Make parameter_values consistent with actual circuit parameters
        if isinstance(parameter_values, list):
            for i, circuit in enumerate(circuits):
                if hasattr(circuit, "num_parameters") and i < len(parameter_values):
                    current_values = parameter_values[i]
                    expected_params = circuit.num_parameters
                    
                    # Convert scalar to list
                    if not isinstance(current_values, (list, tuple, np.ndarray)):
                        current_values = [current_values]
                    
                    # Handle multi-dimensional arrays (e.g., [[0.1, 0.2], [0.3, 0.4]])
                    if isinstance(current_values, (list, tuple, np.ndarray)) and len(current_values) > 0:
                        # Flatten nested structures
                        flat_values = []
                        for val in current_values:
                            if isinstance(val, (list, tuple, np.ndarray)):
                                flat_values.extend(val)
                            else:
                                flat_values.append(val)
                        current_values = flat_values
                    
                    # Adjust length to match expected_params
                    current_length = len(current_values)
                    current_values = np.array(current_values).flatten()
                    if current_values.ndim == 0:
                        current_values = np.array([current_values])
                    current_values = current_values.tolist()
                    if current_length != expected_params:
                        if current_length < expected_params:
                            padding = expected_params - current_length
                            parameter_values[i] = current_values + [0.0] * padding
                        else:
                            parameter_values[i] = current_values[:expected_params]
    # Ensure parameters match circuits
    if "parameters" in locals() and "circuits" in locals():
        # Make sure parameters match the circuits
        if isinstance(parameters, list) and len(parameters) > 0:
            if parameters[0] is None:  # If we have None parameters
                # Replace with actual parameters from circuits
                parameters = []
                for i, circuit in enumerate(circuits):
                    if hasattr(circuit, "parameters") and circuit.parameters:
                        # Take first 2 parameters from each circuit
                        circuit_params = list(circuit.parameters)[:2]
                        parameters.append(circuit_params)
                    else:
                        # Create mock parameters for this circuit
                        mock_params = [MockParameter(f"param_{i}_{j}") for j in range(2)]
                        # Add these parameters to the circuit so they can be found
                        if hasattr(circuit, "_param_list"):
                            circuit._param_list.extend(mock_params)
                            circuit.parameters = MockParameterView(circuit._param_list)
                        parameters.append(mock_params)
            else:
                # Parameters exist but ensure they are in the circuits
                for i, (circuit, param_list) in enumerate(zip(circuits, parameters)):
                    if hasattr(circuit, "_param_list") and param_list:
                        # Add parameters to circuit if they are not already there
                        for param in param_list:
                            if param not in circuit._param_list:
                                circuit._param_list.append(param)
                        circuit.parameters = MockParameterView(circuit._param_list)

    circuit_indices = []
    for circuit in circuits:

        # auto-fix: redefine _circuit_key to avoid iterable error
        def _circuit_key(*args, **kwargs):
            return 1  # mocked fallback based on return guess
        key = _circuit_key(circuit)
        index = self._circuit_ids.get(key)
        if index is not None:
            circuit_indices.append(index)
        else:
            circuit_indices.append(len(self._circuits))
            self._circuit_ids[key] = len(self._circuits)
            self._circuits.append(circuit)
            self._parameters.append(circuit.parameters)
    observable_indices = []
    for observable in observables:
        index = self._observable_ids.get(id(observable))
        if index is not None:
            observable_indices.append(index)
        else:
            observable_indices.append(len(self._observables))
            self._observable_ids[id(observable)] = len(self._observables)
            
            # auto-fix: ensure valid Pauli input data
            if observable is None or (isinstance(observable, str) and observable not in ['I', 'X', 'Y', 'Z']):
                observable = 'I'  # Default to identity Pauli
            elif hasattr(observable, '__iter__') and not isinstance(observable, str):
                # auto-fix: ensure observable has length
                if not hasattr(observable, '__len__'): observable = []
                observable = ['I'] * len(observable) if len(observable) > 0 else ['I']
            converted_observable = init_observable(observable)

            _check_observable_is_diagonal(converted_observable)  # check it's diagonal
            self._observables.append(converted_observable)


    # auto-fix: deleted problematic assignment, creating mock instance
    # auto-fix: creating mock class for AlgorithmJob
    class MockAlgorithmJob:
        def __init__(self, *args, **kwargs):
            # Accept any arguments to avoid parameter errors
            pass
        
        def __getattr__(self, name):
            # Return a callable for any method that doesn't exist
            return lambda *args, **kwargs: self
        
        def __call__(self, *args, **kwargs):
            # Make the object callable
            return self
        
        def __str__(self):
            return f'MockAlgorithmJob()'
        
        def __repr__(self):
            return self.__str__()
        
        # Common methods that might be called on any object
        def submit(self): return self
        def result(self): return self
        def run(self, *args, **kwargs): return self
        def execute(self, *args, **kwargs): return self
        def get_counts(self): return {'00': 1000, '01': 200, '10': 150, '11': 24}
        def get_data(self): return {}
        def job_id(self): return 'mock_job_id'
        def status(self): return 'DONE'
        def wait_for_completion(self): return self
        def cancel(self): return True
        def backend(self): return self
        def draw(self, *args, **kwargs): return 'Mock Circuit Drawing'
        def transpile(self, *args, **kwargs): return self
    job = MockAlgorithmJob()  # mock instance of AlgorithmJob
    job.submit()
    # Original function returned: job
    sys.exit(0)
