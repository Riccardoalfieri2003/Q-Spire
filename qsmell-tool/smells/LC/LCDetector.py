import ast
from smells.Detector import Detector
from smells.LC.LC import LC

































































@Detector.register(LC)
class LCDetector(Detector, ast.NodeVisitor):

    smell_cls = LC

    def __init__(self, smell_cls):
        super().__init__(smell_cls)
        self.calls = []
        self.assignments = {}
        self.matrixes = []  # <-- added: will store unique concrete matrixes





    def detect(self, code: str) -> list[LC]:
        smells = []

        
