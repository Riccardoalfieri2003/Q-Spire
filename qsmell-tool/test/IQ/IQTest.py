from smells.Detector import Detector
from smells.IQ.IQDetector import IQDetector
from smells.IQ.IQ import IQ




def test_custom_gate_detector():
    """
        Test the custom gate detector with example code.
    
        Make sure to be inside the folder QSmell_Tool\qsmell-tool
        SiIQe imports are relative, in order to test the code below execute the following script in the terminal

        python -m test.IQ.IQTest
        
    """
    
    # Example quantum code with custom gates
    file="test/IQ/IQCode.py"
    """with open("test/IQ/IQCode.py", "r") as file:
        test_code = file.read()    """


    
    detector = Detector(IQ)
    smel=detector.detect(file)

    for smell in smel:
        print(smell.as_dict())

if __name__ == "__main__":
    test_custom_gate_detector()