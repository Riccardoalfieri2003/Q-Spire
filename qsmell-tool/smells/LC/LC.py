from smells.QuantumSmell import QuantumSmell

class LC(QuantumSmell):
    def __init__(self, likelihood: float, error: dict, lenght_op: float, parallel_op: float, 
                 backend: str = None, circuit_name: str = None, 
                 explanation: str = None, suggestion: str = None,
                 circuit:dict = None):
        super().__init__(
            type_="LC",
            explanation=explanation,
            suggestion=suggestion,
            circuit_name=circuit_name,
            circuit=circuit
        )
        self.likelihood = likelihood
        self.error = error
        self.lenght_op = lenght_op
        self.parallel_op = parallel_op
        self.backend = backend

    def update_likelihood(self, likelihood: float):
        self.likelihood = likelihood

    def update_error(self, error: dict):
        if not isinstance(error, dict) or len(error) != 1:
            raise ValueError("Error must be a single-item dictionary")
        self.error = error

    def update_l(self, lenght_op: float):
        self.lenght_op = lenght_op

    def update_c(self, parallel_op: float):
        self.parallel_op = parallel_op

    def update_backend(self, backend: str):
        self.backend = backend

    def as_dict(self):
        base = super().as_dict()
        base.update({
            'likelihood': self.likelihood,
            'error': self.error,
            'lenght_op': self.lenght_op,
            'parallel_op': self.parallel_op,
            'backend': self.backend
        })
        return base