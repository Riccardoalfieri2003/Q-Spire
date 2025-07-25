from smells.OperationCircuitTracker import analyze_quantum_file
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


    def detect(self, file):
        smells = []

        circuits = analyze_quantum_file(file)
        threshold = 2

        for circuit_name, ops in circuits.items():
            qubit_op_indices = defaultdict(list)

            # Collect all operations per qubit
            for index, op in enumerate(ops):
                op_name = op.get('operation_name')
                qubits = op.get('qubits_affected', [])
                row = op.get('row')
                col_start = op.get('column_start')
                col_end = op.get('column_end')

                if not qubits:
                    continue

                for q in qubits:
                    qubit_op_indices[q].append((index, op_name, row, col_start, col_end))

            # Check the distance between first and second operation for each qubit
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

        return smells
