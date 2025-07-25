from smells.OperationCircuitTracker import analyze_quantum_file

"""
python -m test.ROC.ROCTestAlt5
"""

# Example of how to use it
if __name__ == "__main__":
    results=analyze_quantum_file("test/ROC/ROCCode.py", None, debug=False)

    import pprint

    for circuit in results:
        print(circuit)
        pprint.pp(results[circuit])
        print()