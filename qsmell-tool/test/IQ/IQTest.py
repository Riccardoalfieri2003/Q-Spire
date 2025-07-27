from smells.Detector import Detector
from smells.IQ.IQDetector import IQDetector
from smells.IQ.IQ import IQ




def test_detector():
    """
        Test the custom gate detector with example code.
    
        Make sure to be inside the folder QSmell_Tool\qsmell-tool
        SiIQe imports are relative, in order to test the code below execute the following script in the terminal

        python -m test.IQ.IQTest
        
    """
    
    file="test/IQ/IQCode.py"
    
    detector = Detector(IQ)
    smel=detector.detect(file)

    for smell in smel:
        print(smell.as_dict())

if __name__ == "__main__":
    test_detector()