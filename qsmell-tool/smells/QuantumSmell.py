class QuantumSmell:

    def __init__(self, type_: str, row: int = None, column_start: int = None,  column_end: int = None, explanation=None, suggestion=None, circuit_name=None, circuit: dict=None):
        self.type = type_
        self.row = row
        self.column_start = column_start
        self.column_end = column_end
        self.explanation = explanation
        self.suggestion = suggestion
        self.circuit_name = circuit_name 
        self.circuit=circuit

    def set_row(self, row: str):
        self.row = row

    def set_column_start(self, column_start: str):
        self.column_start = column_start

    def set_column_end(self, column_end: str):
        self.column_end = column_end

    def set_explanation(self, explanation: str):
        self.explanation = explanation

    def set_suggestion(self, suggestion: str):
        self.suggestion = suggestion

    def set_circuit_name(self, circuit_name: str):
        self.circuit_name = circuit_name

    def set_circuit(self, circuit:dict):
        self.circuit=circuit

    def as_dict(self):
        return {
            'type': self.type,
            'row': self.row,
            'column_start': self.column_start,
            'column_end': self.column_end,
            'explanation': self.explanation,
            'suggestion': self.suggestion,
            'circuit_name': self.circuit_name,
            'circuit': self.circuit
        }
