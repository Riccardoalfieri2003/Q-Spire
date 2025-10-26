import ast
from smells.Detector import Detector
from smells.LPQ.LPQ import LPQ
from smells.utils.config_loader import get_detector_option

@Detector.register(LPQ)
class LPQDetector(Detector, ast.NodeVisitor):

    smell_cls = LPQ

    def __init__(self, smell_cls):
        super().__init__(smell_cls)
        self.transpile_calls = []
        self.transpile_aliases = set(["transpile"])  # starts with the real name

    def detect(self, file: str) -> list[LPQ]:

        with open(file, "r", encoding="utf-8") as file:
            code = file.read()    

        # Parse and visit AST
        tree = ast.parse(code)
        self.visit(tree)

        smells = []

        for call in self.transpile_calls:
            has_initial_layout = any(
                keyword.arg == "initial_layout" for keyword in call.keywords
            )

            if not has_initial_layout:
                row = call.lineno
                col_start = call.col_offset+1
                col_end = getattr(call, 'end_col_offset', col_start + len("transpile"))+1

                circuit_name = None
                if call.args:
                    first_arg = call.args[0]
                    if isinstance(first_arg, ast.Name):
                        circuit_name = first_arg.id
                    elif isinstance(first_arg, ast.Attribute):
                        circuit_name = first_arg.attr
                    elif isinstance(first_arg, ast.Str):
                        circuit_name = first_arg.s

                smell = LPQ(
                    row=row,
                    col_start=col_start,
                    col_end=col_end,
                    circuit_name=circuit_name,
                    explanation="",
                    suggestion=""
                )
                smells.append(smell)

        min_num_smells = get_detector_option("LPQ", "min_num_smells", fallback=1)
        if len(smells)>=min_num_smells: return smells
        else: return []







        

    def visit_Assign(self, node: ast.Assign):
        """
        Detect assignments like: transpile_alt = transpile
        """
        # Only handle single assignment targets
        if len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                if isinstance(node.value, ast.Name) and node.value.id in self.transpile_aliases:
                    # Found alias
                    self.transpile_aliases.add(target.id)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """
        Check if the called function is named transpile or is an alias.
        """
        called_func_name = None

        if isinstance(node.func, ast.Name):
            called_func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            called_func_name = node.func.attr

        if called_func_name in self.transpile_aliases:
            self.transpile_calls.append(node)

        self.generic_visit(node)
