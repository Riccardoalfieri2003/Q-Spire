# expected: IM on qubit 0 (old Qiskit API names, circuit size taken from a test attribute)
import unittest
from qiskit import QuantumCircuit


class TestOld(unittest.TestCase):
    def setUp(self):
        self.n = 2

    def test_old_api(self):
        qc = QuantumCircuit(self.n, 1)
        qc.u3(0.1, 0.2, 0.3, 0)
        qc.cnot(0, 1)
        qc.measure(0, 0)
        qc.u1(0.5, 0)
        self.assertEqual(qc.size(), 4)
