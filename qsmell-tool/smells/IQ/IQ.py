from smells.QuantumSmell import QuantumSmell

class IQ(QuantumSmell):

    def __init__(self, row: int, column_start: int, column_end: int, operation_distance: int, qubit: int, operation_name: str,
                 explanation=None, suggestion=None, circuit_name=None):
        super().__init__(
            type_='IQ',
            row=row,
            column_start=column_start,
            column_end=column_end,
            explanation=explanation,
            suggestion=suggestion,
            circuit_name=circuit_name
        )
        self.operation_distance = operation_distance
        self.qubit=qubit
        self.operation_name=operation_name

    def set_operation_distance(self, distance: int):
        self.operation_distance = distance

    def as_dict(self):
        base_dict = super().as_dict()
        base_dict.update({
            'qubit': self.qubit,
            'operation_distance': self.operation_distance,
            'operation_name': self.operation_name
        })
        return base_dict