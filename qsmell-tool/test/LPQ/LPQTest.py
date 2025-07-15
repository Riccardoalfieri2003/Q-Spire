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
    
    # Example quantum code with custom gates
    with open("test/LPQ/LPQCode.py", "r", encoding="utf-8") as file:
        test_code = file.read()    


    
    detector = Detector(LPQ)
    smel=detector.detect(test_code)

    for smell in smel:
        print(smell.as_dict())

if __name__ == "__main__":
    test_custom_gate_detector()