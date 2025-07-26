from smells.Detector import Detector
from smells.LPQ.LPQDetector import LPQDetector
from smells.LPQ.LPQ import LPQ




def test_custom_gate_detector():
    """
        Test the custom gate detector with example code.
    
        Make sure to be inside the folder QSmell_Tool\qsmell-tool
        Since imports are relative, in order to test the code below execute the following script in the terminal

        python -m test.LPQ.LPQTest
        
    """

    file="test/LPQ/LPQCode.py"
    
    detector = Detector(LPQ)
    smel=detector.detect(file)

    for smell in smel:
        print(smell.as_dict())

if __name__ == "__main__":
    test_custom_gate_detector()