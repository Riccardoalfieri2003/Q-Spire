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
    
    detector = Detector(CG)
    smel=detector.detect(file)



    for smell in smel:
        explanation = Explainer.explain(smell)

        print(explanation)

        
        
        break
        
    


if __name__ == "__main__":
    test_detector()