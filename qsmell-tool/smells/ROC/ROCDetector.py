import ast
from pathlib import Path
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from collections import defaultdict
from smells.Detector import Detector
from smells.ROC.ROC import ROC
from smells.IM.IMInstrumentation_post import EnhancedQubitTracker
from smells.IM.IMInstrumentation_pre import CircuitAutoTracker


@Detector.register(ROC)
class ROCDetector(Detector):

    def __init__(self, smell_cls):
        super().__init__(smell_cls)
        self.circuit_tracker = None
        self.operation_tracker = None
        self.position_tracker = None
    
    """
    
    def detect(self, original_code):

        smells = []

        # Read instrumentation segments
        with open("smells/IM/IMInstrumentation_pre.py", "r", encoding="utf-8") as file:
            pre_code = file.read()
        with open("smells/IM/IMInstrumentation_post.py", "r", encoding="utf-8") as file:
            post_code = file.read()

        code_segments = [
            "# --- PRE-INSTRUMENTATION CODE ---",
            pre_code,
            "# --- ORIGINAL CODE ---",
            original_code,
            "# --- POST-INSTRUMENTATION CODE ---",
            post_code
        ]
        combined_code = '\n'.join(code_segments)

        exec_namespace = {
            'QuantumRegister': QuantumRegister,
            'ClassicalRegister': ClassicalRegister,
            'QuantumCircuit': QuantumCircuit,
            'defaultdict': defaultdict,
            '__ORIGINAL_CODE__': original_code  # Inject original code for position tracking
        }

        try:
            exec(combined_code, exec_namespace)
            self.position_tracker = exec_namespace.get('position_tracker')

        except Exception as e:
            print("Error executing combined code. Code was:")
            print(combined_code)
            raise RuntimeError(f"Execution error: {str(e)}") from e
    """
        

    def detect(self, original_code):

        smells = []

        # Read instrumentation segments
        with open("smells/IM/IMInstrumentation_pre.py", "r", encoding="utf-8") as file:
            pre_code = file.read()
        with open("smells/IM/IMInstrumentation_post.py", "r", encoding="utf-8") as file:
            post_code = file.read()

        post_post_code = """

# build AST positions
circuit_tracker.find_operations_with_positions(__ORIGINAL_CODE__)

# now track each circuit, writing into circuit_tracker.trackers
for name in circuit_tracker.all_circuits:
    qc = globals()[name]
    if qc is None:
        continue

    # build a simple position-lookup map of (ln,c0,c1) → (ln,c0,c1)
    pos_map = {
        (ln, c0, c1): (ln, c0, c1)
        for _, _, ln, c0, c1 in circuit_tracker.positions[name]
    }

    tracker = EnhancedQubitTracker()
    tracker.track(qc, positions=pos_map)
    # now this attribute exists
    circuit_tracker.trackers[name] = tracker
"""

        code_segments = [
            "# --- PRE-INSTRUMENTATION CODE ---",
            pre_code,
            "# --- ORIGINAL CODE ---",
            original_code,
            "# --- POST-INSTRUMENTATION CODE ---",
            post_code,
            post_post_code
        ]
        combined_code = '\n'.join(code_segments)

        """
        exec_namespace = {
            'QuantumRegister': QuantumRegister,
            'ClassicalRegister': ClassicalRegister,
            'QuantumCircuit': QuantumCircuit,
            'defaultdict': defaultdict,
            '__ORIGINAL_CODE__': original_code  # Inject original code for position tracking
        }

        print(combined_code)
        """

        # 2) Create the namespace dict and seed it
        exec_namespace = {
            '__ORIGINAL_CODE__': original_code,
            'QuantumCircuit': QuantumCircuit,
            #'Position': Position,                   # if you use Position() in post code
            'CircuitAutoTracker': CircuitAutoTracker,
            'EnhancedQubitTracker': EnhancedQubitTracker,
            # plus any other imports your code references...
        }

        """
        # 3) Exec all three blocks into that namespace
        exec(combined_code, exec_namespace)

        # 4) Pull out your results from the namespace
        circuit_tracker   = exec_namespace['circuit_tracker']
        positions_map     = circuit_tracker.get_all_positions()
        operations_map    = {
            name: tracker.get_sequential_operations()
            for name, tracker in circuit_tracker.trackers.items()
        }
        """

        try:
            exec(combined_code, exec_namespace)
            self.position_tracker = exec_namespace.get('position_tracker')

            #print(self.position_tracker)

            circuit_tracker = exec_namespace['circuit_tracker']
            positions_map  = circuit_tracker.positions  # this is your flatten map

            print(circuit_tracker)
            print()
            print(positions_map)

            """
            # -------------- ROC DETECTION STARTS HERE ----------------
            threshold = 3  # configurable idle length
            for circuit_name, ops in self.position_tracker.items():
                last_use = {}  # maps qubit index -> last op index

                for op_index, (op_name, qubits, row, col_start, col_end) in enumerate(ops):
                    # Skip empty qubit operations (like barriers or measure_all)
                    if not qubits:
                        continue

                    for q in range(0, max(qubits) + 1):  # handle all qubits seen so far
                        if q not in last_use and q in qubits:
                            # First time this qubit is used — skip
                            last_use[q] = op_index 
                            continue

                        if q not in qubits:
                            # Qubit not used in this op, check distance
                            if q in last_use:
                                distance = op_index - last_use[q]
                                if distance >= threshold:
                                    prev_op = ops[op_index]
                                    smell = ROC(
                                        row=prev_op[2],
                                        column_start=prev_op[3],
                                        column_end=prev_op[4],
                                        qubit=q,
                                        operation_distance=distance,
                                        circuit_name=circuit_name
                                    )
                                    smells.append(smell)
                                    last_use[q] = op_index  # Update last use to prevent duplicates

                    # Update last_use for qubits used in this op
                    for q in qubits:
                        last_use[q] = op_index
            """

            threshold = 3
            for circuit_name, ops in positions_map.items():
                last_op_index = {}
                for index, (op_name, qubits, row, col_start, col_end) in enumerate(ops):
                    if not qubits:
                        continue
                    for q in qubits:
                        if q in last_op_index:
                            distance = index - last_op_index[q]
                            if distance > threshold:
                                smell = ROC(
                                    row=row,
                                    column_start=col_start,
                                    column_end=col_end,
                                    circuit_name=circuit_name,
                                    qubit=q,
                                    operation_distance=distance,
                                    operation_name=op_name
                                )
                                smells.append(smell)
                        last_op_index[q] = index


            """
            print("\n--- Qubit Operation Distances ---")
            for circuit_name, ops in self.position_tracker.items():
                print(f"Circuit: {circuit_name}")
                last_use = {}
                for op_index, (op_name, qubits, row, col_start, col_end) in enumerate(ops):
                    if not qubits:
                        continue
                    for q in qubits:
                        if q in last_use:
                            distance = op_index - last_use[q]
                            print(f"Qubit {q} | Op '{op_name}' at line {row} | Distance from last use: {distance}")
                        else:
                            print(f"Qubit {q} | Op '{op_name}' at line {row} | First use (no previous op)")
                        last_use[q] = op_index
            """
            

            # -------------- ROC DETECTION ENDS HERE ----------------

        except Exception as e:
            print("Error executing combined code. Code was:")
            print(combined_code)
            raise RuntimeError(f"Execution error: {str(e)}") from e

        return smells
