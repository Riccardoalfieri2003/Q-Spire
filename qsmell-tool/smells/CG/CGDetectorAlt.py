import ast
from smells.Detector import Detector
from smells.CG.CG import CG


"""
@Detector.register(CG)
class CGDetector:
    smell_class = CG

    def detect(self, code: str) -> list[CG]:
        smells = []

        class UnitaryCallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.calls = []
                self.unitary_names = {"unitary"}           # functions or aliases: unitary(...)
                self.unitary_gate_aliases = {"UnitaryGate"}
                self.unitary_gate_instances = set()
                self.obj_attr_aliases = set()              # e.g., {"unitary_alt"}

            def visit_ImportFrom(self, node):
                for alias in node.names:
                    if alias.name == "unitary":
                        self.unitary_names.add(alias.asname or alias.name)
                    elif alias.name == "UnitaryGate":
                        self.unitary_gate_aliases.add(alias.asname or alias.name)
                self.generic_visit(node)

            def visit_Assign(self, node):
                # Track: my_alias = unitary
                if isinstance(node.value, ast.Name) and node.value.id in self.unitary_names:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.unitary_names.add(target.id)

                # Track: unitary_alt = QuantumCircuit.unitary
                if isinstance(node.value, ast.Attribute):
                    if node.value.attr == "unitary":
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.obj_attr_aliases.add(target.id)

                # Track: gate = UnitaryGate(...)
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name):
                        if node.value.func.id in self.unitary_gate_aliases:
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    self.unitary_gate_instances.add(target.id)

                self.generic_visit(node)

            def visit_Call(self, node):
                is_unitary_call = False

                if isinstance(node.func, ast.Name):
                    # direct call: unitary(...) or alias
                    if node.func.id in self.unitary_names:
                        is_unitary_call = True

                elif isinstance(node.func, ast.Attribute):
                    attr = node.func.attr

                    # obj.unitary()
                    if attr == "unitary":
                        is_unitary_call = True

                    # obj.unitary_alt()
                    elif attr in self.obj_attr_aliases:
                        is_unitary_call = True

                    # obj.append(gate, ...)
                    elif attr == "append" and node.args:
                        if isinstance(node.args[0], ast.Name):
                            if node.args[0].id in self.unitary_gate_instances:
                                is_unitary_call = True

                if is_unitary_call:
                    self.calls.append({
                        "row": node.lineno,
                        "col": node.col_offset,
                    })

                self.generic_visit(node)

            def visit_BinOp(self, node):
                # Handle: qc << UnitaryGate(...)
                if isinstance(node.op, ast.LShift):
                    if isinstance(node.right, ast.Call):
                        if isinstance(node.right.func, ast.Name):
                            if node.right.func.id in self.unitary_gate_aliases:
                                self.calls.append({
                                    "row": node.lineno,
                                    "col": node.col_offset,
                                })
                self.generic_visit(node)


        tree = ast.parse(code)
        visitor = UnitaryCallVisitor()
        visitor.visit(tree)

        for call_info in visitor.calls:
            smells.append(self.smell_class(
                row=call_info["row"],
                col=call_info["col"] + 1,
                explanation=None,
                suggestion=None
            ))

        return smells
"""

"""
import ast
from smells.Detector import Detector
from smells.CG.CG import CG

@Detector.register(CG)
class CGDetector:
    smell_class = CG

    def detect(self, code: str) -> list[CG]:
        smells = []

        class UnitaryCallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.calls = []
                self.unitary_names = {"unitary"}
                self.unitary_gate_aliases = {"UnitaryGate"}
                self.unitary_gate_instances = set()
                self.obj_attr_aliases = set()
                self.variables = {}   # <-- to track simple vars

            def visit_ImportFrom(self, node):
                for alias in node.names:
                    if alias.name == "unitary":
                        self.unitary_names.add(alias.asname or alias.name)
                    elif alias.name == "UnitaryGate":
                        self.unitary_gate_aliases.add(alias.asname or alias.name)
                self.generic_visit(node)

            def visit_Assign(self, node):
                if isinstance(node.value, (ast.List, ast.Tuple, ast.Constant, ast.Call)):
                    try:
                        value = ast.literal_eval(node.value)
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.variables[target.id] = value
                    except Exception:
                        pass

                # Existing logic for aliases:
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
                    if isinstance(node.value.func, ast.Name):
                        if node.value.func.id in self.unitary_gate_aliases:
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    self.unitary_gate_instances.add(target.id)

                self.generic_visit(node)

            def visit_Call(self, node):
                is_unitary_call = False
                circuit_name = None
                matrix = None
                qubits = None

                if isinstance(node.func, ast.Attribute):
                    attr = node.func.attr
                    if attr == "unitary" or attr in self.obj_attr_aliases:
                        is_unitary_call = True
                    elif attr == "append" and node.args:
                        if isinstance(node.args[0], ast.Name):
                            if node.args[0].id in self.unitary_gate_instances:
                                is_unitary_call = True
                    if isinstance(node.func.value, ast.Name):
                        circuit_name = node.func.value.id

                elif isinstance(node.func, ast.Name):
                    if node.func.id in self.unitary_names:
                        is_unitary_call = True

                if is_unitary_call:
                    # Find matrix & qubits:
                    if len(node.args) >= 2:
                        matrix_node = node.args[0]
                        qubits_node = node.args[1]

                        # If list directly
                        if isinstance(qubits_node, (ast.List, ast.Tuple)):
                            try:
                                qubits = ast.literal_eval(qubits_node)
                            except Exception:
                                pass
                        elif isinstance(qubits_node, ast.Name):
                            qubits = self.variables.get(qubits_node.id)

                        if isinstance(matrix_node, ast.Name):
                            matrix = self.variables.get(matrix_node.id)
                        else:
                            try:
                                matrix = ast.literal_eval(matrix_node)
                            except Exception:
                                pass

                    self.calls.append({
                        "row": node.lineno,
                        "col": node.col_offset,
                        "circuit_name": circuit_name,
                        "matrix": matrix,
                        "qubits": qubits,
                    })

                self.generic_visit(node)

            def visit_BinOp(self, node):
                # Optional: handle qc << UnitaryGate(...)
                if isinstance(node.op, ast.LShift):
                    if isinstance(node.right, ast.Call):
                        if isinstance(node.right.func, ast.Name):
                            if node.right.func.id in self.unitary_gate_aliases:
                                circuit_name = None
                                if isinstance(node.left, ast.Name):
                                    circuit_name = node.left.id
                                self.calls.append({
                                    "row": node.lineno,
                                    "col": node.col_offset,
                                    "circuit_name": circuit_name,
                                    "matrix": None,
                                    "qubits": None,
                                })
                self.generic_visit(node)


        tree = ast.parse(code)
        visitor = UnitaryCallVisitor()
        visitor.visit(tree)

        for call_info in visitor.calls:
            smells.append(self.smell_class(
                row=call_info["row"],
                col=call_info["col"] + 1,
                explanation=None,
                suggestion=None,
                circuit_name=call_info.get("circuit_name"),
                matrix=call_info.get("matrix"),
                qubits=call_info.get("qubits")
            ))

        return smells
"""





import ast
from smells.Detector import Detector
from smells.CG.CG import CG

@Detector.register(CG)
class CGDetector:
    smell_class = CG

    def detect(self, code: str) -> list[CG]:
        smells = []

        class UnitaryCallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.calls = []
                self.unitary_names = {"unitary"}
                self.unitary_gate_aliases = {"UnitaryGate"}
                self.unitary_gate_instances = set()
                self.obj_attr_aliases = set()
                self.variables = {}

            def visit_ImportFrom(self, node):
                for alias in node.names:
                    if alias.name == "unitary":
                        self.unitary_names.add(alias.asname or alias.name)
                    elif alias.name == "UnitaryGate":
                        self.unitary_gate_aliases.add(alias.asname or alias.name)
                self.generic_visit(node)

            def visit_Assign(self, node):
                if isinstance(node.value, (ast.List, ast.Tuple, ast.Constant, ast.Call)):
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
                    if isinstance(node.value.func, ast.Name):
                        if node.value.func.id in self.unitary_gate_aliases:
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    self.unitary_gate_instances.add(target.id)

                self.generic_visit(node)

            def visit_Call(self, node):
                is_unitary_call = False
                circuit_name = None
                matrix = None
                qubits = None

                if isinstance(node.func, ast.Attribute):
                    attr = node.func.attr
                    if attr == "unitary" or attr in self.obj_attr_aliases:
                        is_unitary_call = True
                    elif attr == "append" and node.args:
                        if isinstance(node.args[0], ast.Name):
                            if node.args[0].id in self.unitary_gate_instances:
                                is_unitary_call = True
                    if isinstance(node.func.value, ast.Name):
                        circuit_name = node.func.value.id

                elif isinstance(node.func, ast.Name):
                    if node.func.id in self.unitary_names:
                        is_unitary_call = True

                if is_unitary_call:
                    if len(node.args) >= 2:
                        matrix_node = node.args[0]
                        qubits_node = node.args[1]

                        if isinstance(qubits_node, (ast.List, ast.Tuple)):
                            try:
                                qubits = ast.literal_eval(qubits_node)
                            except Exception:
                                pass
                        elif isinstance(qubits_node, ast.Name):
                            qubits = self.variables.get(qubits_node.id)

                        if isinstance(matrix_node, ast.Name):
                            matrix = self.variables.get(matrix_node.id)
                        else:
                            try:
                                matrix = ast.literal_eval(matrix_node)
                            except Exception:
                                pass

                    # compute column_end
                    col_start = node.col_offset
                    col_end = getattr(node, "end_col_offset", col_start + 1)  # fallback: +1

                    self.calls.append({
                        "row": node.lineno,
                        "col_start": col_start,
                        "col_end": col_end,
                        "circuit_name": circuit_name,
                        "matrix": matrix,
                        "qubits": qubits,
                    })

                self.generic_visit(node)

            def visit_BinOp(self, node):
                if isinstance(node.op, ast.LShift):
                    if isinstance(node.right, ast.Call):
                        if isinstance(node.right.func, ast.Name):
                            if node.right.func.id in self.unitary_gate_aliases:
                                circuit_name = None
                                if isinstance(node.left, ast.Name):
                                    circuit_name = node.left.id
                                col_start = node.col_offset
                                col_end = getattr(node, "end_col_offset", col_start + 1)
                                self.calls.append({
                                    "row": node.lineno,
                                    "col_start": col_start,
                                    "col_end": col_end,
                                    "circuit_name": circuit_name,
                                    "matrix": None,
                                    "qubits": None,
                                })
                self.generic_visit(node)

        tree = ast.parse(code)
        visitor = UnitaryCallVisitor()
        visitor.visit(tree)

        for call_info in visitor.calls:
            smells.append(self.smell_class(
                row=call_info["row"],
                col_start=call_info["col_start"] + 1,   # make 1-based
                col_end=call_info["col_end"] + 1,       # make 1-based
                matrix=call_info.get("matrix"),
                qubits=call_info.get("qubits"),
                circuit_name=call_info.get("circuit_name"),
                explanation=None,
                suggestion=None
            ))

        return smells