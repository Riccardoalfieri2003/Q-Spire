# expected: nothing (a transpiler pass run on a DAG is not a circuit execution)
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import Optimize1qGates


def optimize():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.x(0)
    dag = circuit_to_dag(qc)
    out = Optimize1qGates().run(dag)
    print(out)
