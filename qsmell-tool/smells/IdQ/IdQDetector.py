from smells.utils.OperationCircuitTracker import analyze_quantum_file
from smells.Detector import Detector
from smells.IdQ.IdQ import IdQ


@Detector.register(IdQ)
class IdQDetector(Detector):

    def __init__(self, smell_cls):
        super().__init__(smell_cls)
        self.circuit_tracker = None
        self.operation_tracker = None
        self.position_tracker = None
    
    
    def detect(self, file):
        smells = []

        circuits = analyze_quantum_file(file)
        threshold = 3

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
