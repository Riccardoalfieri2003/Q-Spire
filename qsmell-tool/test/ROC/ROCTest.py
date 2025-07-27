from smells.Detector import Detector
from smells.ROC.ROCDetector import ROCDetector
from smells.ROC.ROC import ROC

def test_detector():
    """
        Test the custom gate detector with example code.
    
        Make sure to be inside the folder QSmell_Tool\qsmell-tool
        Since imports are relative, in order to test the code below execute the following script in the terminal

        python -m test.ROC.ROCTest
        
    """
    
    file="test/ROC/ROCCode.py"
    
    detector = Detector(ROC)
    smel=detector.detect(file)

    for smell in smel:
        print(smell.as_dict())

if __name__ == "__main__":
    test_detector()