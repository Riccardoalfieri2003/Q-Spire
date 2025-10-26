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
from detection.DynamicDetection.GeneralFileTest import detect_smells_from_file


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

    #print()

    """# Print results
    for pyFile in pyFiles:
        if pyFile in smells and smells[pyFile]:  # Only print if we have results
            print(f"Smells in {pyFile}:")
            for smell in smells[pyFile]: 
                print(smell.as_dict())
            print()"""

    return smells




import os
import csv

def save_output(path: str, folder_name: str, smells: dict):
    """Save detected smells to CSV files in a designated folder."""
    
    # Filter out files that have at least one smell
    files_with_smells = {file: smell_list for file, smell_list in smells.items() if smell_list}

    if not files_with_smells:
        print("No smells detected — no output folder created.")
        return

    # Create the full output directory
    output_folder = os.path.join(path, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving smells to: {output_folder}")

    for file_path, smell_list in files_with_smells.items():
        # Extract base name of the Python file (without path) to use as CSV filename
        base_filename = os.path.basename(file_path)
        filename_wo_ext = os.path.splitext(base_filename)[0]
        csv_file_path = os.path.join(output_folder, f"{filename_wo_ext}.csv")

        # Convert all smells to dictionaries
        dict_rows = [smell.as_dict() for smell in smell_list]

        if dict_rows:
            # Get all unique keys from all smell dicts to define CSV header
            fieldnames = sorted({k for d in dict_rows for k in d.keys()})

            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(dict_rows)

            print(f"Saved {len(dict_rows)} smells to {csv_file_path}")




if __name__ == "__main__":

    # python -m detection.DynamicDetection.GeneralFolderTest

    folder = os.path.abspath("C:/Users/rical/OneDrive/Desktop/alt_def")
    smells = detect_smells_from_folder(folder)

    # Example: save results to desktop under folder "SmellResults"
    output_path = os.path.abspath("C:/Users/rical/OneDrive/Desktop")
    output_folder_name = "SmellResults"

    save_output(output_path, output_folder_name, smells)
