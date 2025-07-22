import ast
from pathlib import Path
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from collections import defaultdict
from smells.Detector import Detector
from smells.IdQ.IdQ import IdQ


@Detector.register(IdQ)
class IdQDetector(Detector):

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

            #print(self.position_tracker)

            """
            # -------------- IDQ DETECTION STARTS HERE ----------------
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
                                    smell = IdQ(
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
            for circuit_name, ops in self.position_tracker.items():
                last_op_index = {}
                for index, (op_name, qubits, row, col_start, col_end) in enumerate(ops):
                    if not qubits:
                        continue
                    for q in qubits:
                        if q in last_op_index:
                            distance = index - last_op_index[q]
                            if distance > threshold:
                                smell = IdQ(
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
            

            # -------------- IDQ DETECTION ENDS HERE ----------------

        except Exception as e:
            print("Error executing combined code. Code was:")
            print(combined_code)
            raise RuntimeError(f"Execution error: {str(e)}") from e

        return smells
