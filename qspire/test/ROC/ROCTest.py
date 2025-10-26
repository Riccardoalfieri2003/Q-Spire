from smells.Detector import Detector
from smells.ROC.ROCDetector import ROCDetector
from smells.ROC.ROC import ROC

from smells.Explainer import Explainer
from smells.ROC.ROCExplainer import ROCExplainer

def test_detector():
    """
        Test the custom gate detector with example code.
    
        Make sure to be inside the folder QSmell_Tool\qspire
        Since imports are relative, in order to test the code below execute the following script in the terminal

        python -m test.ROC.ROCTest
        
    """
    
    file="test/ROC/ROCCode.py"
    
    with open(file, "r") as f:
        code = f.read()

    detector = Detector(ROC)
    smells=detector.detect(file)



    for smell in smells:
        explanation_generator = Explainer.explain(code, smell, 'dynamic')
        
        if explanation_generator:
            for chunk in explanation_generator:  # Iterate over the generator
                print(chunk, end="", flush=True)
            print()  # New line after each explanation
        
        
        break

if __name__ == "__main__":
    test_detector()