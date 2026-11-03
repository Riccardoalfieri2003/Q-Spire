# expected: nothing (parameters bound through the sampler PUB)
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorSampler


def sample():
    theta = Parameter("t")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    qc.measure_all()
    sampler = StatevectorSampler()
    job = sampler.run([(qc, [0.1])])
    print(job.result()[0].data.meas.get_counts())
