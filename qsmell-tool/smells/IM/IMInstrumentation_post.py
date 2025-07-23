"""from smells.IM.IMInstrumentation_pre import CircuitAutoTracker, EnhancedQubitTracker

# Set up circuit tracker
circuit_tracker = CircuitAutoTracker()
circuit_tracker.find_circuits_in_globals(globals())  # Detect circuits
circuit_tracker.track_all_circuits(globals())        # Track operations

# Extract operation sequence
operation_tracker = circuit_tracker.get_all_operations()

# Parse original code for position info
position_tracker = None
if '__ORIGINAL_CODE__' in globals():
    circuit_tracker.find_operations_with_positions(globals()['__ORIGINAL_CODE__'])
    position_tracker = circuit_tracker.get_all_positions()"""




# --- POST‑INSTRUMENTATION CODE ---
from qiskit.circuit import Instruction
from smells.IM.IMInstrumentation_pre import CircuitAutoTracker

class Position:
    def __init__(self, lineno, col_offset, end_col_offset):
        self.lineno = lineno
        self.col_offset = col_offset
        self.end_col_offset = end_col_offset

class EnhancedQubitTracker:
    def __init__(self):
        self.global_order = []

    def track(self, circuit, positions=None):
        for instr, qargs, _ in circuit.data:
            # 1) repeat → empty qubits
            if instr.name == 'repeat':
                pos = self._lookup(instr, positions)
                self.global_order.append((
                    'repeat', [],
                    pos.lineno if pos else None,
                    pos.col_offset if pos else None,
                    pos.end_col_offset if pos else None
                ))
                continue

            # 2) inline composite (append) definitions
            if isinstance(instr, Instruction) and instr.definition is not None:
                # map each subcircuit qubit → parent qubit
                mapping = dict(zip(instr.definition.qubits, qargs))
                pos = self._lookup(instr, positions)
                for def_instr, def_qargs, _ in instr.definition.data:
                    parents   = [ mapping[sq] for sq in def_qargs ]
                    idxs      = [ circuit.qubits.index(q) for q in parents ]
                    self.global_order.append((
                        def_instr.name,
                        idxs,
                        pos.lineno if pos else None,
                        pos.col_offset if pos else None,
                        pos.end_col_offset if pos else None
                    ))
                continue

            # 3) simple gate/barrier/etc.
            name = instr.name
            idxs = [circuit.qubits.index(q) for q in qargs]
            self.global_order.append((name, idxs, None, None, None))

        return self

    def _lookup(self, instr, positions):
        if not positions: return None
        key = (
            getattr(instr, '_directive_lineno', None),
            getattr(instr, '_directive_col_offset', None),
            getattr(instr, '_directive_end_col_offset', None)
        )
        return positions.get(key)

    def get_sequential_operations(self):
        return self.global_order


# === usage in your detector after exec() ===

# 1) build AST positions first
circuit_tracker = CircuitAutoTracker()

"""
# 1) Build the flattened AST positions
circuit_tracker.find_operations_with_positions(__ORIGINAL_CODE__)

# 2) Now track each QuantumCircuit
for name in circuit_tracker.all_circuits:
    qc = globals()[name]  # picks up the actual QC object built in ORIGINAL block
    if qc is None:
        continue
    # Build a quick lookup: (lineno, col0, col1) -> Position(...)
    pos_map = {
        (ln, c0, c1): Positions(ln, c0, c1)
        for op, _, ln, c0, c1 in circuit_tracker.positions[name]
    }
    tracker = EnhancedQubitTracker()
    tracker.track(qc, positions=pos_map)
    circuit_tracker.trackers[name] = tracker"""