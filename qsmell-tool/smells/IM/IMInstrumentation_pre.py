# --- PRE‑INSTRUMENTATION CODE ---
import ast
from collections import defaultdict
from qiskit import QuantumCircuit

class CircuitAutoTracker:
    def __init__(self):
        # detected circuit names → None for now
        self.all_circuits = {}
        # circuit_name → list of (op, [qubits], lineno, col0, col1)
        self.positions    = defaultdict(list)
        # storage for per‑circuit trackers
        self.trackers     = {}     

    def find_circuits_in_globals(self, global_vars):
        for name, obj in global_vars.items():
            if isinstance(obj, QuantumCircuit):
                self.all_circuits[name] = None

    def find_operations_with_positions(self, source_code):
        tree = ast.parse(source_code)
        circuit_names = set(self.all_circuits.keys())
        classical_registers = set()
        quantum_register_sizes = {}

        # 1) register size inference
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                func = getattr(node.value.func, 'id', None)
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        if func == 'ClassicalRegister':
                            classical_registers.add(tgt.id)
                        elif func == 'QuantumRegister' and node.value.args:
                            arg0 = node.value.args[0]
                            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, int):
                                quantum_register_sizes[tgt.id] = arg0.value

        # 2) collect raw positions
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and hasattr(node.func, 'value'):
                circ = getattr(node.func.value, 'id', None)
                if circ in circuit_names:
                    op = getattr(node.func, 'attr', None)
                    qubits = []
                    params = []
                    args = node.args

                    # parametric ops
                    if op in ['rz','rx','ry','u','p'] and args:
                        first = args[0]
                        if isinstance(first, (ast.Constant, ast.BinOp, ast.UnaryOp, ast.Call)):
                            try:
                                params.append(ast.unparse(first))
                                args = args[1:]
                            except Exception:
                                pass

                    # qubit indices
                    for arg in args:
                        if isinstance(arg, ast.Subscript):
                            reg = getattr(arg.value, 'id', None)
                            idx = None
                            if isinstance(arg.slice, ast.Constant):
                                idx = arg.slice.value
                            elif hasattr(arg.slice, 'value') and isinstance(arg.slice.value, ast.Constant):
                                idx = arg.slice.value.value
                            if reg and idx is not None and reg not in classical_registers:
                                qubits.append(idx)
                        elif isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                            qubits.append(arg.value)
                        elif isinstance(arg, ast.Name):
                            reg = arg.id
                            if reg not in classical_registers:
                                size = quantum_register_sizes.get(reg)
                                if size is not None:
                                    qubits.extend(range(size))
                                else:
                                    qubits.append(reg)

                    if params:
                        op = f"{op}({','.join(params)})"

                    self.positions[circ].append((
                        op, qubits,
                        node.lineno,
                        node.col_offset,
                        getattr(node, 'end_col_offset', None)
                    ))

        # 3) flatten out all `append` entries
        new_pos = defaultdict(list)
        for circ, ops in self.positions.items():
            for op, args, ln, c0, c1 in ops:
                if op == 'append':
                    sub_name   = args[0]
                    qubit_args = args[1:]
                    for sub_op, sub_qs, *_ in self.positions[sub_name]:
                        mapped = [ qubit_args[i] for i in sub_qs ]
                        new_pos[circ].append((sub_op, mapped, ln, c0, c1))
                else:
                    new_pos[circ].append((op, args, ln, c0, c1))
        self.positions = new_pos

    def get_all_positions(self):
        return dict(self.positions)

    # no-op stub; your post-code will fill self.trackers directly
    def track_all_circuits(self, global_vars):
        pass
