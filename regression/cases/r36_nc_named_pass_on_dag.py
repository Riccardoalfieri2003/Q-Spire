# expected: nothing (a transpiler pass from a missing package run on a DAG is not an execution)
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from some_missing_package.passes import CheckDirection


def check():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    dag = circuit_to_dag(qc)
    pass_ = CheckDirection(None)
    pass_.run(dag)
    print(pass_)
