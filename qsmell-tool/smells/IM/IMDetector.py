import ast
from pathlib import Path
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from collections import defaultdict
from smells.Detector import Detector
from smells.IM.IM import IM

"""
@Detector.register(IM)
class IMDetector(Detector):

    def __init__(self, smell_cls):
        super().__init__(smell_cls)
        self.circuit_tracker = None
        self.operation_tracker = None
    
    def detect(self, original_code, source_file=None):
        #Execute all code segments together with proper __file__ handling
        # Read pre and post instrumentation code from files
        with open("smells/IM/IMInstrumentation_pre.py", "r", encoding="utf-8") as file:
            pre_code = file.read()
        
        with open("smells/IM/IMInstrumentation_post.py", "r", encoding="utf-8") as file:
            post_code = file.read()

        # Create execution namespace with required variables
        exec_namespace = {
            'QuantumRegister': QuantumRegister,
            'ClassicalRegister': ClassicalRegister,
            'QuantumCircuit': QuantumCircuit,
            'defaultdict': defaultdict,
            'ast': ast,
            '__file__': source_file if source_file else "<string>",  # Add this line
            '__name__': '__main__'
        }
        
        # Combine code segments
        combined_code = '\n'.join([
            "# --- PRE-INSTRUMENTATION CODE ---",
            pre_code,
            "# --- ORIGINAL CODE ---",
            original_code,
            "# --- POST-INSTRUMENTATION CODE ---",
            post_code
        ])


        #print(combined_code)
        
        try:
            exec(combined_code, exec_namespace)
            #self.circuit_tracker = exec_namespace.get('circuit_tracker')
            self.operation_tracker = exec_namespace.get('operation_tracker')

            print(f"[DEBUG] Operations: {self.operation_tracker}")
            return self

        except Exception as e:
            print(f"Execution error: {e}")
            raise

    def _execute_code_from_file(self, file_path):
        #Execute Python code from a file
        file_content = Path(file_path).read_text()
        exec(file_content, globals())
"""


import ast
from pathlib import Path
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from collections import defaultdict
from smells.Detector import Detector
from smells.IM.IM import IM

def find_classical_registers(self, source_code):
    tree = ast.parse(source_code)
    self.classical_registers = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                if hasattr(node.value.func, 'id') and node.value.func.id == 'ClassicalRegister':
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.classical_registers.add(target.id)

@Detector.register(IM)
class IMDetector(Detector):

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
