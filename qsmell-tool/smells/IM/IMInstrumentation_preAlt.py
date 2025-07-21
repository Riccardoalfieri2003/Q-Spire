import ast
from collections import defaultdict
from qiskit import QuantumCircuit

class CircuitAutoTracker:
    def __init__(self):
        self.all_circuits = {}
        self.trackers = {}
        
    def find_circuits_in_file(self, filename):
        """Parse Python file and find all QuantumCircuit instances"""
        with open(filename, 'r') as f:
            tree = ast.parse(f.read())
            
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and \
               isinstance(node.value, ast.Call) and \
               hasattr(node.value.func, 'id') and \
               node.value.func.id == 'QuantumCircuit' ):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                    self.all_circuits[var_name] = None  # Placeholder

    def track_all_circuits(self, global_vars):
        """Track all found circuits using their variable names"""
        for name in self.all_circuits:
            if name in global_vars and isinstance(global_vars[name], QuantumCircuit):
                tracker = EnhancedQubitTracker()
                tracker.track(global_vars[name])
                self.trackers[name] = tracker
                
    def get_all_operations(self):
        """Return operations for all tracked circuits"""
        return {name: tracker.get_sequential_operations() 
                for name, tracker in self.trackers.items()}
    
    def find_circuits_in_globals(self, global_vars):
        """Find QuantumCircuit instances directly from globals"""
        for name, obj in global_vars.items():
            if isinstance(obj, QuantumCircuit):
                self.all_circuits[name] = None

                

class EnhancedQubitTracker:
    def __init__(self):
        self.qubit_ops = defaultdict(list)  # {qubit: [op1, op2]}
        self.global_order = []              # [(op_name, [qubit_indices]), ...]
    
    def track(self, circuit):
        """Record operations with both per-qubit and global order tracking"""
        for instruction, qargs, _ in circuit.data:
            op_name = instruction.name
            qubit_indices = [self._get_qubit_index(circuit, q) for q in qargs]
            
            # Track per-qubit
            for q in qubit_indices:
                self.qubit_ops[q].append(op_name)
            
            # Track global order
            self.global_order.append((op_name, qubit_indices))
        return self
    
    def _get_qubit_index(self, circuit, qubit):
        return circuit.qubits.index(qubit) if hasattr(circuit, 'qubits') else qubit.index
    
    def get_qubit_operations(self):
        """Returns {qubit: [op1, op2]}"""
        return dict(sorted(self.qubit_ops.items()))
    
    def get_sequential_operations(self):
        """Returns execution sequence as list of (operation, [qubits]) tuples"""
        return [(op, qubits) for op, qubits in self.global_order]

# Example Usage
tracker = EnhancedQubitTracker()