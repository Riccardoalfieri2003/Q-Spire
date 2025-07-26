from smells.OperationCircuitTracker import analyze_quantum_file

"""
python -m test.ROC.ROCTestAlt5
"""
"""
# Example of how to use it
if __name__ == "__main__":
    results=analyze_quantum_file("test/ROC/ROCCode.py", None, debug=False)

    import pprint

    for circuit in results:
        print(circuit)
        pprint.pp(results[circuit])
        print()"""

from smells.Detector import Detector
from smells.ROC.ROCDetector import ROCDetector
from smells.ROC.ROC import ROC




def test_custom_gate_detector():
    """
        Test the custom gate detector with example code.
    
        Make sure to be inside the folder QSmell_Tool\qsmell-tool
        SiROCe imports are relative, in order to test the code below execute the following script in the terminal

        python -m test.ROC.ROCTestAlt5
        
    """
    
    # Example quantum code with custom gates
    file="test/ROC/ROCCode.py"
    
    detector = Detector(ROC)
    smel=detector.detect(file)

    for smell in smel:
        print(smell.as_dict())

if __name__ == "__main__":
    test_custom_gate_detector()




"""
{'type': 'ROC', 'row': None, 'column_start': None, 'column_end': None, 'explanation': '', 'suggestion': '', 'circuit_name': 'qc', 'rows': [], 'operations': [
   ('z', (0,)), 
   ('measure', (0,)), 
   ('rx', (1,)), 
   ('z', (0,)), 
   ('measure', (1,)),
    ('rx', (2,)), 
    ('rx', (2,)), 
    ('z', (0,)), 
    ('measure', (2,)), 
    ('z', (0,)), 
    ('z', (1,)), 
    ('z', (2,)), 
    ('barrier', (0, 1, 2)), 
    ('repeat', (0, 1, 2))], 'repetitions': 0}

"""