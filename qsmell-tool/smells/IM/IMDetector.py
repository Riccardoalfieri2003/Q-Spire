from smells.Detector import Detector
from smells.IM.IM import IM
from smells.utils.OperationCircuitTracker import analyze_quantum_file


@Detector.register(IM)
class IMDetector(Detector):

    def __init__(self, smell_cls):
        super().__init__(smell_cls)
        self.circuit_tracker = None
        self.operation_tracker = None
        self.position_tracker = None

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