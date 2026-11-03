# expected: nothing (parameters bound through coerced SamplerPub objects)
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorSampler
from qiskit.primitives.containers.sampler_pub import SamplerPub


def sample():
    theta = Parameter("t")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    qc.measure_all()
    coerced_pubs = [SamplerPub.coerce((qc, [0.1]))]
    sampler = StatevectorSampler()
    job = sampler.run(coerced_pubs)
    print(job.result()[0].data.meas.get_counts())
