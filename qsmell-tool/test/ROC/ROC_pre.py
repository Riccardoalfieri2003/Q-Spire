import ast

class GeneralQiskitOpVisitor(ast.NodeVisitor):
    def __init__(self, circuit_variable_names=None):
        super().__init__()
        self.operations = []
        self.circuit_variable_names = circuit_variable_names or set()

    def visit_Assign(self, node):
        # Track variables that store QuantumCircuit objects
        if isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "QuantumCircuit":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.circuit_variable_names.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check if it's a method call on a circuit variable
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            circuit_var = node.func.value.id
            if circuit_var in self.circuit_variable_names:
                self.operations.append({
                    "circuit": circuit_var,
                    "method": node.func.attr,  # e.g., h, cx, custom_gate
                    "lineno": node.lineno,
                    "col_offset": node.col_offset,
                    "end_lineno": getattr(node, "end_lineno", node.lineno),
                    "end_col_offset": getattr(node, "end_col_offset", node.col_offset + 1),
                    "raw": ast.unparse(node)  # full source of the op
                })
        self.generic_visit(node)
