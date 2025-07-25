from smells.OperationCircuitTracker import analyze_quantum_file
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


        except Exception as e:
            print("Error executing combined code. Code was:")
            print(combined_code)
            raise RuntimeError(f"Execution error: {str(e)}") from e

        return smells
    """

    
    def detect(self, file):
        smells = []

        circuits = analyze_quantum_file(file)
        threshold = 3

        """for circuit in circuits:
            import pprint
            print(circuit)
            pprint.pp(circuits[circuit])"""

        for circuit_name, ops in circuits.items():
            last_op_index = {}
            for index, op in enumerate(ops):
                op_name = op.get('operation_name')
                qubits = op.get('qubits_affected', [])
                row = op.get('row')
                col_start = op.get('column_start')
                col_end = op.get('column_end')

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

        return smells
