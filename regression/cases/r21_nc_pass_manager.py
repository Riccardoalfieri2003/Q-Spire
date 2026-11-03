# expected: nothing (PassManager.run transpiles, it does not execute)
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import generate_preset_pass_manager


def compile_it():
    backend = GenericBackendV2(num_qubits=3)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    isa = pm.run(qc)
    print(isa.size())
