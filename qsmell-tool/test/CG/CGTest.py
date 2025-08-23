from smells.Detector import Detector
from smells.CG.CGDetector import CGDetector
from smells.CG.CG import CG
from smells.Explainer import Explainer
from smells.CG.CGExplainer import CGExplainer


def test_detector():
    """
        Test the custom gate detector with example code.
    
        Make sure to be inside the folder QSmell_Tool\qsmell-tool
        Since imports are relative, in order to test the code below execute the following script in the terminal

        python -m test.CG.CGTest
        
    """

    file="test/CG/CGCode.py"
    
    with open(file, "r") as f:
        code = f.read()

    detector = Detector(CG)
    smells=detector.detect(file)



    for smell in smells:
        explanation_generator = Explainer.explain(code, smell)
        
        if explanation_generator:
            for chunk in explanation_generator:  # Iterate over the generator
                print(chunk, end="", flush=True)
            print()  # New line after each explanation
        
        
        break
        
    


if __name__ == "__main__":
    test_detector()