from smells.QuantumSmell import QuantumSmell

class LC(QuantumSmell):
    def __init__(self, likelihood: float, error: dict, l: float, c: float, 
                 backend: str = None, circuit_name: str = None, 
                 explanation: str = None, suggestion: str = None):
        super().__init__(
            type_="LC",
            row=None,
            column_start=None,
            column_end=None,
            explanation=explanation,
            suggestion=suggestion,
            circuit_name=circuit_name
        )
        self.likelihood = likelihood
        self.error = error
        self.l = l
        self.c = c
        self.backend = backend

    def update_likelihood(self, likelihood: float):
        self.likelihood = likelihood

    def update_error(self, error: dict):
        if not isinstance(error, dict) or len(error) != 1:
            raise ValueError("Error must be a single-item dictionary")
        self.error = error

    def update_l(self, l: float):
        self.l = l

    def update_c(self, c: float):
        self.c = c

    def update_backend(self, backend: str):
        self.backend = backend

    def as_dict(self):
        base = super().as_dict()
        base.update({
            'likelihood': self.likelihood,
            'error': self.error,
            'l': self.l,
            'c': self.c,
            'backend': self.backend
        })
        return base