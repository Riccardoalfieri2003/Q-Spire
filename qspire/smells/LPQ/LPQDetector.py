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
        self.transpile_aliases = set()      # names bound to qiskit's transpile
        self.qiskit_module_aliases = set()  # names bound to qiskit / qiskit.compiler modules
        self.star_import_from_qiskit = False
        self.locally_defined = set()

    def detect(self, file: str) -> list[LPQ]:

        with open(file, "r", encoding="utf-8") as file:
            code = file.read()    

        # Parse and visit AST
        tree = ast.parse(code)
        # qspire-fix: resolve which names refer to Qiskit's transpile before visiting calls
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "qiskit":
                for alias in node.names:
                    if alias.name == "transpile":
                        self.transpile_aliases.add(alias.asname or alias.name)
                    elif alias.name == "*":
                        self.star_import_from_qiskit = True
                    elif alias.name == "compiler":
                        self.qiskit_module_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "qiskit":
                        self.qiskit_module_aliases.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "transpile":
                self.locally_defined.add(node.name)
        if self.star_import_from_qiskit and "transpile" not in self.locally_defined:
            self.transpile_aliases.add("transpile")
        self.visit(tree)

        smells = []

        for call in self.transpile_calls:
            has_initial_layout = any(
                keyword.arg == "initial_layout" for keyword in call.keywords
            )
            # qspire-fix: with **kwargs the layout may be passed indirectly; do not flag
            if any(keyword.arg is None for keyword in call.keywords):
                continue

            if not has_initial_layout:
                row = call.lineno
                column_start = call.col_offset+1
                column_end = getattr(call, 'end_col_offset', column_start + len("transpile"))+1

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
                    column_start=column_start,
                    column_end=column_end,
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
        # qspire-fix: flag only calls to Qiskit's transpile, i.e., a name imported from qiskit
        # (or an alias of it) or an attribute of a qiskit module (qiskit.transpile,
        # qiskit.compiler.transpile); other functions or methods named transpile are skipped
        if isinstance(node.func, ast.Name):
            if node.func.id in self.transpile_aliases:
                self.transpile_calls.append(node)
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "transpile":
            base = node.func.value
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in self.qiskit_module_aliases:
                self.transpile_calls.append(node)

        self.generic_visit(node)
