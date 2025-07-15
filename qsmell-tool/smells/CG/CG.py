from smells.QuantumSmell import QuantumSmell

class CG(QuantumSmell):
    def __init__(self, row, col_start, col_end, matrix, qubits, circuit_name=None, gate_type=None, explanation=None, suggestion=None):
        super().__init__("CG", row, col_start, col_end, explanation, suggestion, circuit_name)
        self.matrix = matrix
        self.qubits = qubits
        self.gate_type = gate_type

    def update_matrix(self, matrix):
        self.matrix = matrix

    def update_qubits(self, qubits):
        self.qubits = qubits

    def update_gate_type(self, gate_type):
        self.gate_type = gate_type

    def update_explanation(self, explanation):
        self.set_explanation(explanation)

    def update_suggestion(self, suggestion):
        self.set_suggestion(suggestion)

    def as_dict(self):
        base = super().as_dict()
        base.update({
            'matrix': self.matrix,
            'qubits': self.qubits,
            #'gate_type': self.gate_type
        })
        return base
