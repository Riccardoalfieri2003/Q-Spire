from smells.Detector import Detector
from smells.LC.LCDetector import LCDetector
from smells.LC.LC import LC
from smells.Explainer import Explainer
from smells.LC.LCExplainer import LCExplainer

def test_detector():
    """
        Test the custom gate detector with example code.
    
        Make sure to be inside the folder QSmell_Tool\qsmell-tool
        Since imports are relative, in order to test the code below execute the following script in the terminal

        python -m test.LC.LCTest
        
    """

    file="test/LC/LCCode.py"

    with open(file, "r") as f:
        code = f.read()
    
    detector = Detector(LC)
    smells=detector.detect(file)

    for smell in smells:

        print(smell.as_dict())
        break
        explanation_generator = Explainer.explain(code, smell, 'dynamic')
        
        if explanation_generator:
            for chunk in explanation_generator:  # Iterate over the generator
                print(chunk, end="", flush=True)
            print()  # New line after each explanation
        
        break

if __name__ == "__main__":
    test_detector()