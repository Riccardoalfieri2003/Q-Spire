from smells.Detector import Detector
from smells.IM.IMDetector import IMDetector
from smells.IM.IM import IM
from smells.Explainer import Explainer
from smells.IM.IMExplainer import IMExplainer



def test_detector():
    """
        Test the custom gate detector with example code.
    
        Make sure to be inside the folder QSmell_Tool\qsmell-tool
        Since imports are relative, in order to test the code below execute the following script in the terminal

        python -m test.IM.IMTest
        
    """

    file="test/IM/IMCode.py"
        
    with open(file, "r") as f:
        code = f.read()

    detector = Detector(IM)
    smells=detector.detect(file)



    for smell in smells:

        print(smell.as_dict())

        explanation_generator = Explainer.explain(code, smell, 'dynamic')
        
        if explanation_generator:
            for chunk in explanation_generator:  # Iterate over the generator
                print(chunk, end="", flush=True)
            print()  # New line after each explanation
        
        
        break

if __name__ == "__main__":
    test_detector()