from smells.Detector import Detector
from smells.IdQ.IdQDetector import IdQDetector
from smells.IdQ.IdQ import IdQ

from smells.Explainer import Explainer
from smells.IdQ.IdQExplainer import IdQExplainer


def test_detector():
    """
        Test the custom gate detector with example code.
    
        Make sure to be inside the folder QSmell_Tool\qsmell-tool
        Since imports are relative, in order to test the code below execute the following script in the terminal

        python -m test.IdQ.IdQTest
        
    """
    
    file="test/IdQ/IdQCode.py"

    with open(file, "r") as f:
        code = f.read()

    detector = Detector(IdQ)
    smells=detector.detect(file)

    print(type(smells))
    

    for smell in smells:

        print(smell.as_dict())

        

        """explanation_generator = Explainer.explain(code, smell, 'dynamic')
        
        if explanation_generator:
            for chunk in explanation_generator:  # Iterate over the generator
                print(chunk, end="", flush=True)
            print()  # New line after each explanation
        
        
        break"""

if __name__ == "__main__":
    test_detector()