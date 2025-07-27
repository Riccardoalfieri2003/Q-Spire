from smells.Detector import Detector
from smells.NC.NCDetector import NCDetector
from smells.NC.NC import NC




def test_detector():
    """
        Test the custom gate detector with example code.
    
        Make sure to be inside the folder QSmell_Tool\qsmell-tool
        Since imports are relative, in order to test the code below execute the following script in the terminal

        python -m test.NC.NCTest
        
    """

    file="test/NC/NCCode.py"
    
    detector = Detector(NC)
    smel=detector.detect(file)

    for smell in smel:
        import pprint
        pprint.pp(smell.as_dict())
        print()

if __name__ == "__main__":
    test_detector()