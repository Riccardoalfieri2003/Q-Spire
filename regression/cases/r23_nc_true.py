# expected: NC (a circuit executed without any parameter binding)
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def execute_it():
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    sim = AerSimulator()
    r1 = sim.run(qc, shots=10).result()
    r2 = sim.run(qc, shots=20).result()
    print(r1.get_counts(), r2.get_counts())
