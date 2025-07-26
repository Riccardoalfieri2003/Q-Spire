from smells.Detector import Detector
from smells.LC.LCDetector import LCDetector
from smells.LC.LC import LC




def test_custom_gate_detector():
    """
        Test the custom gate detector with example code.
    
        Make sure to be inside the folder QSmell_Tool\qsmell-tool
        Since imports are relative, in order to test the code below execute the following script in the terminal

        python -m test.LC.LCTest
        
    """

    file="test/LC/LCCode.py"
    
    detector = Detector(LC)
    smel=detector.detect(file)

    for smell in smel:
        print(smell.as_dict())

if __name__ == "__main__":
    test_custom_gate_detector()