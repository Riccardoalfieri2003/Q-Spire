from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def init_circuit(theta) :
    qc = QuantumCircuit(5, 1)
    qc.h(0)

    for i in range(4):
        qc.cx(i, i+1)

    qc.barrier()
    qc.rz(theta, range(5) )
    qc.barrier()

    for i in reversed(range (4)):
        qc.cx(i, i+1)

    qc.h(0)
    qc.measure(0, 0)
    return qc

theta_range = [0.00, 0.25, 0.50, 0.75, 1.00]


for theta_val in theta_range:
    qc = init_circuit(theta_val)
    backend = AerSimulator()
    job = backend.run(qc)
    job.result().get_counts()



from qiskit import transpile
from qiskit.circuit import Parameter
theta=Parameter('0')

qc2=init_circuit(theta)

circuits=[qc2.assign_parameters({theta:theta_val}) for theta_val in theta_range]
backend= AerSimulator()
job= backend.run(transpile(circuits,backend))
job.result().get_counts()

job = backend.run(qc2)