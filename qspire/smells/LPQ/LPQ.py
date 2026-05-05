from smells.QuantumSmell import QuantumSmell

class LPQ(QuantumSmell):
    def __init__(self, row, column_start, column_end, circuit_name=None, explanation=None, suggestion=None, circuit=None):
        super().__init__("LPQ", row, column_start, column_end, explanation, suggestion, circuit_name, circuit=circuit)

    def update_explanation(self, explanation):
        self.set_explanation(explanation)

    def update_suggestion(self, suggestion):
        self.set_suggestion(suggestion)

    def as_dict(self):
        base = super().as_dict()
        return base
