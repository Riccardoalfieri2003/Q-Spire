# expected: nothing (run() of a pass manager whose passes come from a missing package)
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from some_missing_provider.passes import ALAPScheduleAnalysis


def schedule():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    pm = PassManager([ALAPScheduleAnalysis(None)])
    out = pm.run(qc)
    out2 = PassManager([ALAPScheduleAnalysis(None)]).run(qc)
    print(out, out2)
