from smells.QuantumSmell import QuantumSmell

class IM(QuantumSmell):
    
    def __init__(self, circuit_name: str, qubit: int, post_measurement_ops: list, 
                 row: int = None, column_start: int = None, column_end: int = None, 
                 explanation: str = None, suggestion: str = None):
        super().__init__(
            type_='IM',
            row=row,
            column_start=column_start,
            column_end=column_end,
            explanation=explanation,
            suggestion=suggestion,
            circuit_name=circuit_name
        )
        self.qubit = qubit
        self.post_measurement_ops = post_measurement_ops

    def set_qubit(self, qubit: int):
        self.qubit = qubit

    def set_post_measurement_ops(self, ops: list):
        self.post_measurement_ops = ops

    def as_dict(self):
        base_dict = super().as_dict()
        base_dict.update({
            'qubit': self.qubit,
            'post_measurement_ops': self.post_measurement_ops
        })
        return base_dict
