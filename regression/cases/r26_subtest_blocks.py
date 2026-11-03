# expected: IM on qubit 0 (circuit built inside `with self.subTest(...)`)
import unittest
from qiskit import QuantumCircuit


class TestSub(unittest.TestCase):
    def test_outcomes(self):
        with self.subTest("normal"):
            qc = QuantumCircuit(2, 1)
            qc.h(0)
            qc.measure(0, 0)
            qc.h(0)
            self.assertEqual(qc.size(), 3)
