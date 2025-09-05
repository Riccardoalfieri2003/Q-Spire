from smells.QuantumSmell import QuantumSmell

class ROC(QuantumSmell):
    def __init__(self, operations, repetitions, rows=None, circuit_name=None, circuit=None):
        # Initialize base QuantumSmell fields with type='ROC' and no row/col info
        super().__init__(
            type_="ROC",
            row=None,
            column_start=None,
            column_end=None,
            explanation="",
            suggestion="",
            circuit_name=circuit_name,
            circuit=circuit
        )
        # ROC-specific attributes
        self.rows = rows if rows is not None else []
        self.operations = operations  # list of operation representations (e.g. strings or op nodes)
        self.repetitions = repetitions

    def set_rows(self, rows):
        self.rows = rows

    def set_operations(self, operations):
        self.operations = operations

    def set_repetitions(self, repetitions):
        self.repetitions = repetitions

    def as_dict(self):
        base_dict = super().as_dict()
        base_dict.update({
            'rows': self.rows,
            'operations': self.operations,
            'repetitions': self.repetitions
        })
        return base_dict
