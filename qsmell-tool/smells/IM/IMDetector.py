from smells.Detector import Detector
from smells.IM.IM import IM
from smells.OperationCircuitTracker import analyze_quantum_file


@Detector.register(IM)
class IMDetector(Detector):

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

            #self.circuit_tracker = exec_namespace.get('circuit_tracker')
            #self.operation_tracker = exec_namespace.get('operation_tracker')
            self.position_tracker = exec_namespace.get('position_tracker')

            #print(self.position_tracker)


            for circuit_name, ops in self.position_tracker.items():

                # Per-qubit list of (operation, full_entry)
                qubit_ops = defaultdict(list)
                for entry in ops:
                    op_name, qubits, row, col_start, col_end = entry
                    for q in qubits:
                        qubit_ops[q].append(entry)
                
                
                #print(qubit_ops)

                # Analyze each qubit
                for qubit, op_list in qubit_ops.items():
                    measure_index = None
                    for i, (op_name, _, row, col_start, col_end) in enumerate(op_list):
                        if op_name == "measure":
                            measure_index = i
                            break  # Only consider first measure for this smell

                    if measure_index is not None and measure_index + 1 < len(op_list):
                        # There are operations after the measure
                        post_ops = []
                        for op in op_list[measure_index + 1:]:
                            post_ops.append(op[0])  # Just the op name
                        
                        # Extract location from the measure op
                        _, _, row, col_start, col_end = op_list[measure_index]

                        # Create the smell
                        smell = IM(
                            circuit_name=circuit_name,
                            qubit=qubit,
                            post_measurement_ops=post_ops,
                            row=row,
                            column_start=col_start,
                            column_end=col_end,
                            explanation="",
                            suggestion=""
                        )
                        smells.append(smell)

            #print(smells)

            return smells

        except Exception as e:
            print("Error executing combined code. Code was:")
            print(combined_code)
            raise RuntimeError(f"Execution error: {str(e)}") from e
    """

    

    def detect(self, file):
        smells = []

        circuits = analyze_quantum_file(file)

        for circuit_name, operations in circuits.items():
            # Dictionary mapping qubit index to list of (index in operations list, operation dict)
            qubit_ops = {}

            for idx, op in enumerate(operations):
                for q in op['qubits_affected']:
                    if q not in qubit_ops:
                        qubit_ops[q] = []
                    qubit_ops[q].append((idx, op))

            # Now check each qubit for non-terminal measurements
            for qubit, op_list in qubit_ops.items():
                for i, (op_idx, op) in enumerate(op_list):
                    if op['operation_name'] == 'measure':
                        # If this is not the last operation on this qubit → it's a smell
                        if i < len(op_list) - 1:
                            smell = IM(
                                circuit_name=circuit_name,
                                qubit=qubit,
                                row=op.get('row'),
                                column_start=op.get('column_start'),
                                column_end=op.get('column_end'),
                                explanation="",
                                suggestion=""
                            )
                            smells.append(smell)

        return smells