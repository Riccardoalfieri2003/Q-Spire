# expected: LPQ (Qiskit transpile without initial_layout), twice: imported name and module attribute
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2


def compile_it():
    backend = GenericBackendV2(num_qubits=3)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    a = transpile(qc, backend)
    b = qiskit.transpile(qc, backend)
    c = transpile(qc, backend, initial_layout=[0, 1])
    print(a.size(), b.size(), c.size())
