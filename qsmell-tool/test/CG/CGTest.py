from smells.Detector import Detector
from smells.CG.CGDetectorAlt import CGDetector
from smells.CG.CG import CG




def test_custom_gate_detector():
    """
        Test the custom gate detector with example code.
    
        Make sure to be inside the folder QSmell_Tool\qsmell-tool
        Since imports are relative, in order to test the code below execute the following script in the terminal

        python -m test.CG.CGTest
        
    """
    
    # Example quantum code with custom gates
    with open("test/CG/CGCode.py", "r", encoding="utf-8") as file:
        test_code = file.read()    


    
    detector = Detector(CG)
    smel=detector.detect(test_code)

    for smell in smel:
        print(smell.as_dict())

    """
    cg_smells = detector.detect_custom_gates(test_code)
    
    print("Custom Gate Detection Results:")
    print(f"Custom gates detected: {len(cg_smells)}")
    print(f"Smell detected: {len(cg_smells) > 0}")
    
    print("\nCustom gate smells:")
    for i, smell in enumerate(cg_smells):
        print(f"  Smell {i+1}: {smell.type} at line {smell.row}, column {smell.column}")
        #print(f"    Explanation: {smell.explanation}")
        #print(f"    Suggestion: {smell.suggestion}")
        print()
    """
    
    
    # Show how to get dict representation
    #print("Dict representation of first smell:")
    #if cg_smells:
    #    print(cg_smells[0].as_dict())
    

    """
    # Get detection summary (for backward compatibility)
    summary = detector.get_detection_summary()
    print(f"\nSummary - Function calls: {summary['total_function_calls']}")
    print(f"Circuits created: {summary['circuits_created']}")
    """
    
    #return cg_smells

if __name__ == "__main__":
    test_custom_gate_detector()