# expected: nothing (parameter values passed to run() next to the circuit, as in V1 primitives)
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


class Gradient:
    def run(self, circuits, observables, parameter_values):
        return None


def gradient():
    a = Parameter("a")
    qc = QuantumCircuit(1)
    qc.rx(a, 0)
    grad = Gradient()
    job = grad.run([qc], ["Z"], [[0.5]])
    print(job)
