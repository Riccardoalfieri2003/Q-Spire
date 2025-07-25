from qiskit import QuantumCircuit
from qiskit.circuit.library import unitary
import numpy as np
from qiskit.circuit.library import HamiltonianGate



"""Direct use of unitary(...) in a circuit"""
qc1 = QuantumCircuit(2)
#my_matrix = np.eye(4)
my_matrix=[
    [0,1],
    [1,0]
]

qc1.unitary(my_matrix, [0, 1])





"""Using HamiltonianGate directly with <<"""
qc2 = QuantumCircuit(2)
hamiltonian = np.eye(4)

qc2 << HamiltonianGate(hamiltonian, time=1.0)




"""Instantiating and appending a gate"""

from qiskit.circuit.library import UnitaryGate

qc3 = QuantumCircuit(2)
u = UnitaryGate(np.eye(4))

qc3.append(u, [0, 1])






"""Using an alias for unitary imported from qiskit"""

from qiskit.circuit.library import unitary as custom_unitary

qc4 = QuantumCircuit(2)
matrix = np.eye(4)

custom_unitary(matrix, [0, 1])






"""Using SingleQubitUnitary"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import SingleQubitUnitary
import numpy as np

qc5 = QuantumCircuit(1)
sq_gate = SingleQubitUnitary(np.eye(2))

qc5.append(sq_gate, [0])










"""Indirect: store gate instance in variable then append"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import HamiltonianGate
import numpy as np

qc6 = QuantumCircuit(2)
matrix = np.eye(4)

my_gate = HamiltonianGate(matrix, time=1.0)
qc6.append(my_gate, [0, 1])







"""Using an object attribute as a gate"""


from qiskit import QuantumCircuit
from qiskit.circuit.library import HamiltonianGate
import numpy as np

class MyGates:
    def __init__(self):
        self.unitary = HamiltonianGate(np.eye(4), time=1.0)

qc7 = QuantumCircuit(2)
gates = MyGates()
qc7.append(gates.unitary, [0, 1])







"""Direct function call to unitary with variables"""

from qiskit.circuit.library import unitary
from qiskit import QuantumCircuit
import numpy as np

qc8 = QuantumCircuit(2)
my_matrix = np.eye(4)
qubits = [0, 1]

unitary(my_matrix, qubits)





"""Calls inside functions"""
def func(qc):
    my_gate = HamiltonianGate(np.eye(4), time=1.0)
    qc.append(my_gate, [0, 1])