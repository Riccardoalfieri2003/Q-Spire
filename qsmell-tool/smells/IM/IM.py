from smells.QuantumSmell import QuantumSmell

class IM(QuantumSmell):
    
    def __init__(self, circuit_name: str, qubit: int,
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

    def set_qubit(self, qubit: int):
        self.qubit = qubit
        
    def as_dict(self):
        base_dict = super().as_dict()
        base_dict.update({
            'qubit': self.qubit,
        })
        return base_dict
