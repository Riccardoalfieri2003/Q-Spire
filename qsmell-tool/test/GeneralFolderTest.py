import threading

from smells.Detector import Detector

from smells.CG.CG import CG
from smells.IdQ.IdQ import IdQ
from smells.IM.IM import IM
from smells.IQ.IQ import IQ
from smells.LC.LC import LC
from smells.LPQ.LPQ import LPQ
from smells.NC.NC import NC
from smells.ROC.ROC import ROC

from smells.CG.CGDetector import CGDetector
from smells.IdQ.IdQDetector import IdQDetector
from smells.IM.IMDetector import IMDetector
from smells.IQ.IQDetector import IQDetector
from smells.LC.LCDetector import LCDetector
from smells.LPQ.LPQDetector import LPQDetector
from smells.NC.NCDetector import NCDetector
from smells.ROC.ROCDetector import ROCDetector

import os
from test.GeneralFileTest import detect_smells_from_file


import threading



def get_all_python_files(folder_path):
    """Get all Python files in a folder recursively."""
    python_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                python_files.append(full_path)
    return python_files

def detect_smells_from_folder(folder: str):
    """Detect smells from all Python files in a folder with proper isolation."""
    smells = {}
    pyFiles = get_all_python_files(folder)

    for pyFile in pyFiles:
        print(f"Analyzing {pyFile}")
        try: 
            # Each file gets completely isolated analysis
            result = detect_smells_from_file(pyFile)
            smells[pyFile] = result if result is not None else []
        except Exception as e:
            print(f"Error analyzing {pyFile}: {e}")
            smells[pyFile] = []
            continue

    print()

    # Print results
    for pyFile in pyFiles:
        if pyFile in smells and smells[pyFile]:  # Only print if we have results
            print(f"Smells in {pyFile}:")
            for smell in smells[pyFile]: 
                print(smell.as_dict())
            print()

    return smells

if __name__ == "__main__":
    """python -m test.GeneralFolderTest"""
    folder = os.path.abspath("C:/Users/rical/OneDrive/Desktop/QSmell_Tool/qsmell-tool/test")
    detect_smells_from_folder(folder)



