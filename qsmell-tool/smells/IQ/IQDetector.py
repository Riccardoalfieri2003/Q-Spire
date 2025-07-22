import ast
from pathlib import Path
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from collections import defaultdict
from smells.Detector import Detector
from smells.IQ.IQ import IQ


@Detector.register(IQ)
class IQDetector(Detector):

    def __init__(self, smell_cls):
        super().__init__(smell_cls)
        self.circuit_tracker = None
        self.operation_tracker = None
        self.position_tracker = None


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

            threshold = 2
            # Loop over each circuit in the position tracker
            for circuit_name, ops in self.position_tracker.items():
                # Track operation indices for each qubit
                qubit_op_indices = defaultdict(list)

                # First, collect all operations per qubit
                for index, (op_name, qubits, row, col_start, col_end) in enumerate(ops):
                    if not qubits:
                        continue
                    for q in qubits:
                        qubit_op_indices[q].append((index, op_name, row, col_start, col_end))

                # Now check the distance between the first and second operation for each qubit
                for q, qubit_ops in qubit_op_indices.items():
                    if len(qubit_ops) >= 2:
                        first_index, *_ = qubit_ops[0]
                        second_index, second_op_name, row, col_start, col_end = qubit_ops[1]
                        distance = second_index - first_index

                        if distance >= threshold:
                            smell = IQ(
                                row=row,
                                column_start=col_start,
                                column_end=col_end,
                                circuit_name=circuit_name,
                                qubit=q,
                                operation_distance=distance,
                                operation_name=second_op_name
                            )
                            smells.append(smell)


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
            

            # -------------- IQ DETECTION ENDS HERE ----------------

        except Exception as e:
            print("Error executing combined code. Code was:")
            print(combined_code)
            raise RuntimeError(f"Execution error: {str(e)}") from e

        return smells
