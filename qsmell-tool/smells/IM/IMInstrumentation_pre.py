import ast
from collections import defaultdict
from qiskit import QuantumCircuit

class CircuitAutoTracker:
    def __init__(self):
        self.all_circuits = {}
        self.trackers = {}
        self.positions = defaultdict(list)  # circuit_name -> [(op, [qubits], lineno, col_start, col_end)]
        
    def find_circuits_in_globals(self, global_vars):
        for name, obj in global_vars.items():
            if isinstance(obj, QuantumCircuit):
                self.all_circuits[name] = None
    """
    def find_operations_with_positions(self, source_code):
        tree = ast.parse(source_code)
        circuit_names = set(self.all_circuits.keys())
        classical_registers = set()

        # First pass: find all classical register names
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if hasattr(node.value.func, 'id') and node.value.func.id == 'ClassicalRegister':
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            classical_registers.add(target.id)

        # Second pass: find operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and hasattr(node.func, 'value'):
                if hasattr(node.func.value, 'id') and node.func.value.id in circuit_names:
                    circuit_name = node.func.value.id
                    op_name = getattr(node.func, 'attr', None)
                    qubits = []
                    params = []

                    remaining_args = node.args

                    # Extract parameterized gates
                    if op_name in ['rz', 'rx', 'ry', 'u', 'p'] and remaining_args:
                        first = remaining_args[0]
                        if isinstance(first, (ast.Constant, ast.BinOp, ast.UnaryOp, ast.Call)):
                            try:
                                param = ast.unparse(first)
                            except Exception:
                                param = "param"
                            params.append(param)
                            remaining_args = remaining_args[1:]

                    # Extract qubit arguments
                    for arg in remaining_args:
                        if isinstance(arg, ast.Subscript):
                            # Handle qreg[0], creg[0]
                            reg_name = None
                            index_value = None

                            if isinstance(arg.value, ast.Name):
                                reg_name = arg.value.id

                            if isinstance(arg.slice, ast.Constant):
                                index_value = arg.slice.value
                            elif hasattr(arg.slice, 'value') and isinstance(arg.slice.value, ast.Constant):
                                index_value = arg.slice.value.value

                            if reg_name and index_value is not None:
                                if reg_name not in classical_registers:
                                    qubits.append(index_value)

                        elif isinstance(arg, ast.Constant):
                            if isinstance(arg.value, int):
                                qubits.append(arg.value)

                        elif isinstance(arg, ast.Name):
                            # Skip classical registers
                            if arg.id not in classical_registers:
                                qubits.append(arg.id)

                    # Format parametric op name
                    if params:
                        op_name = f"{op_name}({','.join(params)})"

                    self.positions[circuit_name].append(
                        (op_name, qubits, node.lineno, node.col_offset,
                        getattr(node, 'end_col_offset', None))
                    )
    """

    def find_operations_with_positions(self, source_code):
        tree = ast.parse(source_code)
        circuit_names = set(self.all_circuits.keys())
        classical_registers = set()
        quantum_register_sizes = {}

        # First pass: find all classical and quantum registers
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if hasattr(node.value.func, 'id'):
                    reg_type = node.value.func.id
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            reg_name = target.id
                            if reg_type == 'ClassicalRegister':
                                classical_registers.add(reg_name)
                            elif reg_type == 'QuantumRegister' and node.value.args:
                                # Extract the register size (assumes constant integer size)
                                if isinstance(node.value.args[0], ast.Constant) and isinstance(node.value.args[0].value, int):
                                    quantum_register_sizes[reg_name] = node.value.args[0].value

        # Second pass: find operations
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and hasattr(node.func, 'value'):
                if hasattr(node.func.value, 'id') and node.func.value.id in circuit_names:
                    circuit_name = node.func.value.id
                    op_name = getattr(node.func, 'attr', None)
                    qubits = []
                    params = []

                    remaining_args = node.args

                    # Extract parameterized gates
                    if op_name in ['rz', 'rx', 'ry', 'u', 'p'] and remaining_args:
                        first = remaining_args[0]
                        if isinstance(first, (ast.Constant, ast.BinOp, ast.UnaryOp, ast.Call)):
                            try:
                                param = ast.unparse(first)
                            except Exception:
                                param = "param"
                            params.append(param)
                            remaining_args = remaining_args[1:]

                    # Extract qubit arguments
                    for arg in remaining_args:
                        if isinstance(arg, ast.Subscript):
                            # Handle qreg[0], creg[0]
                            reg_name = None
                            index_value = None

                            if isinstance(arg.value, ast.Name):
                                reg_name = arg.value.id

                            if isinstance(arg.slice, ast.Constant):
                                index_value = arg.slice.value
                            elif hasattr(arg.slice, 'value') and isinstance(arg.slice.value, ast.Constant):
                                index_value = arg.slice.value.value

                            if reg_name and index_value is not None:
                                if reg_name not in classical_registers:
                                    qubits.append(index_value)

                        elif isinstance(arg, ast.Constant):
                            if isinstance(arg.value, int):
                                qubits.append(arg.value)

                        elif isinstance(arg, ast.Name):
                            # If it's a full register name, replace it with list of all its qubit indices
                            reg_name = arg.id
                            if reg_name not in classical_registers:
                                if reg_name in quantum_register_sizes:
                                    reg_size = quantum_register_sizes[reg_name]
                                    qubits.extend(range(reg_size))
                                else:
                                    qubits.append(reg_name)  # fallback

                    # Format parametric op name
                    if params:
                        op_name = f"{op_name}({','.join(params)})"

                    self.positions[circuit_name].append(
                        (op_name, qubits, node.lineno, node.col_offset,
                        getattr(node, 'end_col_offset', None))
                    )


    def track_all_circuits(self, global_vars):
        for name in self.all_circuits:
            if name in global_vars and isinstance(global_vars[name], QuantumCircuit):
                tracker = EnhancedQubitTracker()
                tracker.track(global_vars[name])
                self.trackers[name] = tracker

    def get_all_operations(self):
        return {name: tracker.get_sequential_operations() 
                for name, tracker in self.trackers.items()}

    def get_all_positions(self):
        return dict(self.positions)


class EnhancedQubitTracker:
    def __init__(self):
        self.qubit_ops = defaultdict(list)
        self.global_order = []

    def track(self, circuit):
        for instruction, qargs, _ in circuit.data:
            op_name = instruction.name
            qubit_indices = [self._get_qubit_index(circuit, q) for q in qargs]

            for q in qubit_indices:
                self.qubit_ops[q].append(op_name)

            self.global_order.append((op_name, qubit_indices))
        return self

    def _get_qubit_index(self, circuit, qubit):
        return circuit.qubits.index(qubit) if hasattr(circuit, 'qubits') else qubit.index

    def get_qubit_operations(self):
        return dict(sorted(self.qubit_ops.items()))

    def get_sequential_operations(self):
        return [(op, qubits) for op, qubits in self.global_order]


# Shared trackers for post instrumentation
tracker = EnhancedQubitTracker()
