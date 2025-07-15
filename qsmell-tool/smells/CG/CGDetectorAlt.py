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
                self.unitary_names = {"unitary"}
                #self.unitary_gate_aliases = {"UnitaryGate"}
                self.unitary_gate_aliases = {"UnitaryGate", "HamiltonianGate", "SingleQubitUnitary"}
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
"""




import ast
from smells.Detector import Detector
from smells.CG.CG import CG

@Detector.register(CG)
class CGDetector(Detector, ast.NodeVisitor):

    smell_cls=CG

    def __init__(self, smell_cls):
        super().__init__(smell_cls)
        self.calls = []
        self.assignments = {}


    def detect(self, code: str) -> list[CG]:
        smells = []


        class UnitaryCallVisitor(ast.NodeVisitor):
            def __init__(self):
                self.calls = []

                # Names of functions like 'unitary'
                self.unitary_names = {"unitary"}

                # Names of unitary gate classes
                self.unitary_gate_aliases = {"UnitaryGate", "HamiltonianGate", "SingleQubitUnitary"}

                # Variables assigned as instances of those gates: e.g., gate = HamiltonianGate(...)
                self.unitary_gate_instances = set()

                # Aliases to obj.unitary = something
                self.obj_attr_aliases = set()

                # Track variables assigned to values so we can resolve later
                self.variables = {}

                # NEW: classes having self.unitary = HamiltonianGate(...)
                self.classes_with_unitary_attr = set()

                # NEW: variables assigned to instances of those classes
                self.vars_instance_of_unitary_class = {}

            def visit_ImportFrom(self, node):
                # Track aliases in imports
                for alias in node.names:
                    if alias.name == "unitary":
                        self.unitary_names.add(alias.asname or alias.name)
                    elif alias.name in self.unitary_gate_aliases:
                        self.unitary_gate_aliases.add(alias.asname or alias.name)
                self.generic_visit(node)

            def visit_ClassDef(self, node):
                # Detect classes with self.unitary = HamiltonianGate(...) in __init__
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        for stmt in item.body:
                            if isinstance(stmt, ast.Assign):
                                for target in stmt.targets:
                                    if isinstance(target, ast.Attribute):
                                        if target.attr == "unitary":
                                            if isinstance(stmt.value, ast.Call):
                                                if isinstance(stmt.value.func, ast.Name):
                                                    if stmt.value.func.id in self.unitary_gate_aliases:
                                                        self.classes_with_unitary_attr.add(node.name)
                self.generic_visit(node)

            def visit_Assign(self, node):
                # Track variable values for later resolution
                if isinstance(node.value, (ast.List, ast.Tuple, ast.Constant)):
                    try:
                        value = ast.literal_eval(node.value)
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.variables[target.id] = value
                    except Exception:
                        pass

                # Track aliases: unitary = other_function
                if isinstance(node.value, ast.Name):
                    if node.value.id in self.unitary_names:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.unitary_names.add(target.id)

                # Track object attribute aliases
                if isinstance(node.value, ast.Attribute):
                    if node.value.attr == "unitary":
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.obj_attr_aliases.add(target.id)

                # Track variables assigned to instances of special classes
                if isinstance(node.value, ast.Call):
                    func = node.value.func
                    if isinstance(func, ast.Name):
                        if func.id in self.classes_with_unitary_attr:
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    self.vars_instance_of_unitary_class[target.id] = func.id

                if isinstance(node.value, ast.Call):
                    func = node.value.func
                    class_name = None
                    if isinstance(func, ast.Name):
                        class_name = func.id
                    elif isinstance(func, ast.Attribute):
                        class_name = func.attr  # prende solo il nome senza il modulo
                    
                    if class_name and class_name in self.classes_with_unitary_attr:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.vars_instance_of_unitary_class[target.id] = class_name

                # Track variables assigned to unitary gates: gate = HamiltonianGate(...)
                if isinstance(node.value, ast.Call):
                    if isinstance(node.value.func, ast.Name):
                        if node.value.func.id in self.unitary_gate_aliases:
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    self.unitary_gate_instances.add(target.id)

                    """
                    # Track variables assigned as instances of special classes
                    if node.value.func.id in self.classes_with_unitary_attr:
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                self.vars_instance_of_unitary_class[target.id] = node.value.func.id
                    """
                    
                self.generic_visit(node)

            def visit_Call(self, node):
                is_unitary_call = False
                circuit_name = None
                matrix = None
                qubits = None
                gate_type = None   # NEW: track the type

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
                                if instance_name in self.vars_instance_of_unitary_class:
                                    if arg0.attr == "unitary":
                                        is_unitary_call = True
                                        gate_type = "unitary"

                    if isinstance(node.func.value, ast.Name):
                        circuit_name = node.func.value.id

                elif isinstance(node.func, ast.Name):
                    if node.func.id in self.unitary_names:
                        is_unitary_call = True
                        gate_type = "unitary"

                if is_unitary_call:
                    # Try to extract matrix and qubits if provided
                    if len(node.args) >= 2:
                        matrix_node = node.args[0]
                        qubits_node = node.args[1]

                        # Resolve qubits
                        if isinstance(qubits_node, (ast.List, ast.Tuple)):
                            try:
                                qubits = ast.literal_eval(qubits_node)
                            except Exception:
                                pass
                        elif isinstance(qubits_node, ast.Name):
                            qubits = self.variables.get(qubits_node.id)

                        # Resolve matrix
                        if isinstance(matrix_node, ast.Name):
                            matrix = self.variables.get(matrix_node.id)
                        else:
                            try:
                                matrix = ast.literal_eval(matrix_node)
                            except Exception:
                                pass

                    col_start = node.col_offset
                    col_end = getattr(node, "end_col_offset", col_start + 1)

                    self.calls.append({
                        "row": node.lineno,
                        "col_start": col_start,
                        "col_end": col_end,
                        "circuit_name": circuit_name,
                        "matrix": matrix,
                        "qubits": qubits,
                        "gate_type": gate_type,   # NEW
                    })
                self.generic_visit(node)

            def visit_BinOp(self, node):
                # Detect qc << HamiltonianGate(...)
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
                                    "gate_type": node.right.func.id,  # NEW
                                })
                self.generic_visit(node)

            """
            def visit_Call(self, node):
                is_unitary_call = False
                circuit_name = None
                matrix = None
                qubits = None

                if isinstance(node.func, ast.Attribute):
                    attr = node.func.attr

                    if attr == "unitary" or attr in self.obj_attr_aliases:
                        # circuit.unitary(...) or object.unitary(...)
                        is_unitary_call = True

                    elif attr == "append" and node.args:
                        arg0 = node.args[0]
                        # circuit.append(gate)
                        if isinstance(arg0, ast.Name):
                            if arg0.id in self.unitary_gate_instances:
                                is_unitary_call = True
                        # circuit.append(HamiltonianGate(...))
                        elif isinstance(arg0, ast.Call):
                            if isinstance(arg0.func, ast.Name):
                                if arg0.func.id in self.unitary_gate_aliases:
                                    is_unitary_call = True
                        # circuit.append(x.unitary)
                        elif isinstance(arg0, ast.Attribute):
                            if isinstance(arg0.value, ast.Name):
                                instance_name = arg0.value.id
                                if instance_name in self.vars_instance_of_unitary_class:
                                    if arg0.attr == "unitary":
                                        is_unitary_call = True

                    if isinstance(node.func.value, ast.Name):
                        circuit_name = node.func.value.id

                elif isinstance(node.func, ast.Name):
                    if node.func.id in self.unitary_names:
                        is_unitary_call = True

                if is_unitary_call:
                    # Try to extract matrix and qubits if provided
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

                    col_start = node.col_offset
                    col_end = getattr(node, "end_col_offset", col_start + 1)

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
                # Detect qc << HamiltonianGate(...)
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
            """




        # Parse and visit
        tree = ast.parse(code)
        visitor = UnitaryCallVisitor()
        visitor.visit(tree)

        # Convert found calls to smells
        for call_info in visitor.calls:
            smells.append(self.smell_cls(
                row=call_info["row"],
                col_start=call_info["col_start"] + 1,
                col_end=call_info["col_end"] + 1,
                matrix=call_info.get("matrix"),
                qubits=call_info.get("qubits"),
                circuit_name=call_info.get("circuit_name"),
                gate_type=call_info.get("gate_type"),   # NEW
                explanation=None,
                suggestion=None
            ))

        return smells




"""
import ast

@Detector.register(CG)
class CGDetector(Detector, ast.NodeVisitor):
    def __init__(self, smell_cls):
        super().__init__(smell_cls)
        self.calls = []
        self.assignments = {}

    def detect(self, code):
        tree = ast.parse(code)
        self.visit(tree)
        return self.get_smells()

    def visit_Assign(self, node):
        # Track assignments like my_matrix = np.eye(4)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            var_name = node.targets[0].id
            self.assignments[var_name] = node.value
        self.generic_visit(node)

    def visit_Call(self, node):
        func_name = self._get_func_name(node.func)

        # Handle direct unitary(...) call
        if func_name.endswith(".unitary") or func_name == "unitary":
            matrix_arg = node.args[0] if node.args else None
            matrix_repr = self._resolve_matrix(matrix_arg)
            qubits = self._extract_qubits(node)
            self.calls.append({
                "row": node.lineno,
                "col_start": node.col_offset,
                "col_end": node.col_offset + len(func_name),
                "circuit_name": self._get_circuit_name(node),
                "matrix": matrix_repr,
                "qubits": qubits,
                "gate_type": "Unitary"
            })
        self.generic_visit(node)

    def visit_Expr(self, node):
        # Handle qc << HamiltonianGate(...)
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.LShift):
            gate_call = node.value.right
            if isinstance(gate_call, ast.Call):
                func_name = self._get_func_name(gate_call.func)
                if func_name == "HamiltonianGate":
                    matrix_arg = gate_call.args[0] if gate_call.args else None
                    matrix_repr = self._resolve_matrix(matrix_arg)
                    self.calls.append({
                        "row": node.lineno,
                        "col_start": node.col_offset,
                        "col_end": node.col_offset + 2,
                        "circuit_name": self._get_circuit_name(node),
                        "matrix": matrix_repr,
                        "qubits": None,
                        "gate_type": "HamiltonianGate"
                    })
        self.generic_visit(node)

    def visit_Attribute(self, node):
        self.generic_visit(node)

    def visit(self, node):
        super().visit(node)

    def _resolve_matrix(self, node):
        if isinstance(node, ast.Name):
            var_name = node.id
            assigned = self.assignments.get(var_name)
            if assigned:
                return ast.unparse(assigned).strip()  # return code as string
            else:
                return var_name  # fallback: var name
        elif isinstance(node, ast.Call):
            return ast.unparse(node).strip()
        else:
            return None

    def _extract_qubits(self, node):
        if len(node.args) >= 2:
            qubits_node = node.args[1]
            try:
                return ast.literal_eval(qubits_node)
            except Exception:
                return ast.unparse(qubits_node)
        return None

    def _get_func_name(self, func):
        if isinstance(func, ast.Attribute):
            return func.attr
        elif isinstance(func, ast.Name):
            return func.id
        return ""

    def _get_circuit_name(self, node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    return node.func.value.id
        elif isinstance(node, ast.BinOp):
            if isinstance(node.left, ast.Name):
                return node.left.id
        return None

    def get_smells(self):
        smells = []
        for call_info in self.calls:
            smells.append(self.smell_cls(
                row=call_info["row"],
                col_start=call_info["col_start"]+1,
                col_end=call_info["col_end"]+1,
                matrix=call_info.get("matrix"),
                qubits=call_info.get("qubits"),
                circuit_name=call_info.get("circuit_name"),
                gate_type=call_info.get("gate_type"),
                explanation=None,
                suggestion=None
            ))
        return smells

"""