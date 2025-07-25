from smells.Detector import Detector
from smells.IdQ.IdQDetector import IdQDetector
from smells.IdQ.IdQ import IdQ




def test_custom_gate_detector():
    """
        Test the custom gate detector with example code.
    
        Make sure to be inside the folder QSmell_Tool\qsmell-tool
        SiIdQe imports are relative, in order to test the code below execute the following script in the terminal

        python -m test.IdQ.IdQTest
        
    """
    
    # Example quantum code with custom gates
    file="test/IdQ/IdQCode.py"
    """with open("test/IdQ/IdQCode.py", "r") as file:
        test_code = file.read()    """


    
    detector = Detector(IdQ)
    smel=detector.detect(file)

    for smell in smel:
        print(smell.as_dict())

if __name__ == "__main__":
    test_custom_gate_detector()