from qiskit import QuantumCircuit
from test.LC.MyFakeBackend import MyFakeBackend

backend = MyFakeBackend(
    num_qubits=5,
    noise_settings={
        'x': (35.5e-9, None, 1e-4),  # (duration, None, error_rate)
        'cz': (500e-9, None, 5e-3),
        'measure': (1000e-9, None, 2e-1)
    },
    t1_values=[75e-6, 80e-6, 90e-6, 100e-6, 110e-6],
    t2_values=[120e-6, 130e-6, 140e-6, 150e-6, 160e-6]
)

qc = QuantumCircuit(1)
qc.h(0)

backend.run(qc)


qc2=QuantumCircuit(3)
qc2.h(0)
qc2.x(2)
qc2.cz(0,2)
qc2.cz(0,1)
qc2.cz(1,2)
qc2.x(1)
backend.run(qc2)