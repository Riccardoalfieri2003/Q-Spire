from smells.QuantumSmell import QuantumSmell

class CG(QuantumSmell):
    def __init__(self, row, col_start, col_end, matrix, qubits, circuit_name=None, explanation=None, suggestion=None):
        super().__init__("CG", row, col_start, col_end, explanation, suggestion, circuit_name)
        self.matrix = matrix
        self.qubits = qubits

    
    
    def update_matrix(self, matrix):
        self.set_matrix(matrix)

    def update_qubits(self, qubits):
        self.set_qubits(qubits)



    def update_explanation(self, explanation):
        self.set_explanation(explanation)

    def update_suggestion(self, suggestion):
        self.set_suggestion(suggestion)


    def as_dict(self):
        base = super().as_dict()
        base.update({
            'matrix': self.matrix,
            'qubits': self.qubits
        })
        return base