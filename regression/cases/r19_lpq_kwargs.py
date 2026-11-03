# expected: nothing (initial_layout may come through **kwargs)
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2


def compile_it():
    backend = GenericBackendV2(num_qubits=3)
    opts = {"initial_layout": [0, 1], "optimization_level": 1}
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    out = transpile(qc, backend, **opts)
    print(out.size())
