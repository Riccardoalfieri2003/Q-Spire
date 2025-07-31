import ast
from smells.Detector import Detector
from smells.CG.CG import CG
from smells.utils.config_loader import get_detector_option

def resolve_matrix(node: ast.AST, variables: dict) -> any:

    if isinstance(node, ast.Name):
        val = variables.get(node.id)
        if val is not None:
            if isinstance(val, ast.AST):
                return resolve_matrix(val, variables)
            else:
                return val
        else:
            return None

    elif isinstance(node, (ast.List, ast.Tuple, ast.Constant)):
        try:
            return ast.literal_eval(node)
        except Exception:
            return None

    elif isinstance(node, ast.Call):
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        return f"<Call to {func_name or 'unknown'}>"

    elif isinstance(node, ast.Attribute):
        return f"<Attribute {node.attr}>"

    else:
        return None


@Detector.register(CG)
class CGDetector(Detector, ast.NodeVisitor):

    smell_cls = CG

    def __init__(self, smell_cls):
        super().__init__(smell_cls)
        self.calls = []
        self.assignments = {}
        self.matrixes = []  # <-- added: will store unique concrete matrixes

    def detect(self, file: str) -> list[CG]:

        with open(file, "r", encoding="utf-8") as file:
            code = file.read()    

        smells = []

        class UnitaryCallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.calls = []

                self.unitary_names = {"unitary"}
                self.unitary_gate_aliases = {"UnitaryGate", "HamiltonianGate", "SingleQubitUnitary"}
                self.unitary_gate_instances = set()
                self.obj_attr_aliases = set()
                self.variables = {}
                self.classes_with_unitary_attr = set()
                self.vars_instance_of_unitary_class = {}

                self.found_matrixes = []  # <-- added: store matrixes found inside visitor

            def visit_ImportFrom(self, node: ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "unitary":
                        self.unitary_names.add(alias.asname or alias.name)
                    elif alias.name in self.unitary_gate_aliases:
                        self.unitary_gate_aliases.add(alias.asname or alias.name)
                self.generic_visit(node)

            def visit_ClassDef(self, node: ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        for stmt in item.body:
                            if isinstance(stmt, ast.Assign):
                                for target in stmt.targets:
                                    if isinstance(target, ast.Attribute) and target.attr == "unitary":
                                        if isinstance(stmt.value, ast.Call):
                                            if isinstance(stmt.value.func, ast.Name):
                                                if stmt.value.func.id in self.unitary_gate_aliases:
                                                    self.classes_with_unitary_attr.add(node.name)
                self.generic_visit(node)

            def visit_Assign(self, node: ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.variables[target.id] = node.value

                if isinstance(node.value, (ast.List, ast.Tuple, ast.Constant)):
                    try:
                        value = ast.literal_eval(node.value)
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.variables[target.id] = value
                    except Exception:
                        pass

                if isinstance(node.value, ast.Name):
                    if node.value.id in self.unitary_names:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.unitary_names.add(target.id)

                if isinstance(node.value, ast.Attribute):
                    if node.value.attr == "unitary":
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.obj_attr_aliases.add(target.id)

                if isinstance(node.value, ast.Call):
                    func = node.value.func
                    class_name = None
                    if isinstance(func, ast.Name):
                        class_name = func.id
                    elif isinstance(func, ast.Attribute):
                        class_name = func.attr

                    if class_name and class_name in self.classes_with_unitary_attr:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.vars_instance_of_unitary_class[target.id] = class_name

                    if isinstance(func, ast.Name):
                        if func.id in self.unitary_gate_aliases:
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    self.unitary_gate_instances.add(target.id)

                self.generic_visit(node)

            def visit_Call(self, node: ast.Call):
                is_unitary_call = False
                circuit_name = None
                matrix = None
                qubits = None
                gate_type = None

                if isinstance(node.func, ast.Attribute):
                    attr = node.func.attr

                    if attr == "unitary" or attr in self.obj_attr_aliases:
                        is_unitary_call = True
                        gate_type = "unitary"

                    elif attr == "append" and node.args:
                        arg0 = node.args[0]
                        if isinstance(arg0, ast.Name):
                            if arg0.id in self.unitary_gate_instances:
                                is_unitary_call = True
                                gate_type = "unitary"
                        elif isinstance(arg0, ast.Call):
                            if isinstance(arg0.func, ast.Name):
                                if arg0.func.id in self.unitary_gate_aliases:
                                    is_unitary_call = True
                                    gate_type = arg0.func.id
                        elif isinstance(arg0, ast.Attribute):
                            if isinstance(arg0.value, ast.Name):
                                instance_name = arg0.value.id
                                if instance_name in self.vars_instance_of_unitary_class and arg0.attr == "unitary":
                                    is_unitary_call = True
                                    gate_type = "unitary"

                    if isinstance(node.func.value, ast.Name):
                        circuit_name = node.func.value.id

                elif isinstance(node.func, ast.Name):
                    if node.func.id in self.unitary_names:
                        is_unitary_call = True
                        gate_type = "unitary"

                if is_unitary_call:
                    if len(node.args) >= 2:
                        matrix_node = node.args[0]
                        qubits_node = node.args[1]

                        matrix = resolve_matrix(matrix_node, self.variables)

                        # <-- added: store found matrix if it's concrete and not placeholder
                        if matrix is not None and not isinstance(matrix, str):
                            self.found_matrixes.append(matrix)

                        if isinstance(qubits_node, (ast.List, ast.Tuple)):
                            try:
                                qubits = ast.literal_eval(qubits_node)
                            except Exception:
                                pass
                        elif isinstance(qubits_node, ast.Name):
                            qubits = self.variables.get(qubits_node.id)

                    col_start = node.col_offset
                    col_end = getattr(node, "end_col_offset", col_start + 1)

                    self.calls.append({
                        "row": node.lineno,
                        "col_start": col_start,
                        "col_end": col_end,
                        "circuit_name": circuit_name,
                        "matrix": matrix,
                        "qubits": qubits,
                        "gate_type": gate_type,
                    })

                self.generic_visit(node)

            def visit_BinOp(self, node: ast.BinOp):
                if isinstance(node.op, ast.LShift):
                    if isinstance(node.right, ast.Call):
                        if isinstance(node.right.func, ast.Name):
                            if node.right.func.id in self.unitary_gate_aliases:
                                circuit_name = None
                                if isinstance(node.left, ast.Name):
                                    circuit_name = node.left.id

                                matrix = None
                                if len(node.right.args) > 0:
                                    matrix = resolve_matrix(node.right.args[0], self.variables)

                                if matrix is not None and not isinstance(matrix, str):
                                    self.found_matrixes.append(matrix)

                                col_start = node.col_offset
                                col_end = getattr(node, "end_col_offset", col_start + 1)
                                self.calls.append({
                                    "row": node.lineno,
                                    "col_start": col_start,
                                    "col_end": col_end,
                                    "circuit_name": circuit_name,
                                    "matrix": matrix,
                                    "qubits": None,
                                    "gate_type": node.right.func.id,
                                })
                self.generic_visit(node)

        # Parse and visit
        tree = ast.parse(code)
        visitor = UnitaryCallVisitor()
        visitor.visit(tree)

        # <-- after visiting: deduplicate matrixes and store in self
        self.matrixes = list({str(m): m for m in visitor.found_matrixes}.values())

        # Convert found calls to smells
        for call_info in visitor.calls:
            smells.append(self.smell_cls(
                row=call_info["row"],
                col_start=call_info["col_start"] + 1,
                col_end=call_info["col_end"] + 1,
                matrix=call_info.get("matrix"),
                qubits=call_info.get("qubits"),
                circuit_name=call_info.get("circuit_name"),
                gate_type=call_info.get("gate_type"),
                explanation=None,
                suggestion=None
            ))

        min_num_smells = get_detector_option("CG", "min_num_smells", fallback=1)
        if len(smells)>=min_num_smells: return smells
        else: return []
