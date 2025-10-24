import ast
import re
import sys
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import importlib.util
import copy

















def extract_quantum_circuits(filepath: str, debug: bool = False) -> Dict[str, Any]:
    """Execute a file as a script and extract all Qiskit quantum circuits from it."""
    
    import runpy
    from qiskit import QuantumCircuit
    
    try:
        # Execute the file as a script (respects if __name__ == "__main__")
        module_dict = runpy.run_path(filepath, run_name="__main__")
        if debug:
            print(f"File executed successfully as script")
    except Exception as e:
        print(f"Error executing file: {e}")
        return {}
    
    if debug:
        print(f"All variables in executed file: {[attr for attr in module_dict.keys() if not attr.startswith('__')]}")
    
    # Extract all Qiskit circuits
    circuits = {}
    
    for var_name, obj in module_dict.items():
        # Skip private/magic variables
        if var_name.startswith('__'):
            continue
        
        try:
            # Check if it's a QuantumCircuit instance
            if isinstance(obj, QuantumCircuit):
                circuits[var_name] = obj
                if debug:
                    print(f"Found QuantumCircuit: {var_name} (qubits: {obj.num_qubits})")
            
            # Also check for lists/tuples that might contain circuits
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    if isinstance(item, QuantumCircuit):
                        circuits[f"{var_name}[{i}]"] = item
                        if debug:
                            print(f"Found QuantumCircuit in {var_name}[{i}] (qubits: {item.num_qubits})")
            
            # Also check for dicts that might contain circuits
            elif isinstance(obj, dict):
                for key, item in obj.items():
                    if isinstance(item, QuantumCircuit):
                        circuits[f"{var_name}['{key}']"] = item
                        if debug:
                            print(f"Found QuantumCircuit in {var_name}['{key}'] (qubits: {item.num_qubits})")
        
        except Exception as e:
            if debug:
                print(f"Error checking {var_name}: {e}")
            continue
    
    if debug:
        print(f"\nTotal circuits found: {len(circuits)}")
        print(f"Circuit names: {list(circuits.keys())}")
    
    # Update circuit sizes
    
    return circuits



def analyze_quantum_file_circuits(input_file: str, output_file: str = None, debug: bool = False):
    """
    Analyze a quantum circuit file and optionally save results.
    
    Args:
        input_file: Path to Python file containing quantum circuits
        output_file: Optional path to save analysis results
        debug: If True, print debugging information
    """
    #analyzer = QuantumCircuitAnalyzer()
    #results = analyzer.analyze_file(input_file, debug)

    results= extract_quantum_circuits(input_file, debug=True)
    
    # Save to file if requested
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

    return results