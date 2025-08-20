import csv
from pathlib import Path
import shutil
import sys
import threading
from detection.StaticDetection.StaticCircuit import FunctionExecutionGenerator
from detection.StaticDetection.AutoFix import auto_fix
import os
from detection.DynamicDetection.GeneralFileTest import detect_smells_from_file

from detection.StaticDetection.StaticMappedDetection import autofix_map_detect


def get_all_python_files(folder_path):
    """Get all Python files in a folder recursively."""
    python_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                python_files.append(full_path)
    return python_files



import os
import csv


def save_output(output_saving_folder, smells, folder):
    os.makedirs(output_saving_folder, exist_ok=True)

    for file_path, smell_list in smells.items():
        # Skip if no smells
        if not smell_list:
            continue

        # Encode full path into a valid filename
        safe_name = file_path.replace(folder,"").replace(":", "_").replace("\\", "_").replace("/", "_")
        output_file_path = os.path.join(output_saving_folder, f"{safe_name}.csv")

        # Convert all smells to dict form
        smells_dicts = [s.as_dict() for s in smell_list]

        # Collect all unique fieldnames across all smells
        all_keys = set()
        for d in smells_dicts:
            all_keys.update(d.keys())
        fieldnames = sorted(all_keys)  # sorted for consistency

        with open(output_file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in smells_dicts:
                writer.writerow(d)



def detect_smells_from_folder(folder):
    smells={}
    try:
        pyFiles = get_all_python_files(folder)
        for file in pyFiles:
            smells[file]=autofix_map_detect(file)
    except: pass

    return smells


"""
if __name__ == "__main__":
    # python -m detection.StaticDetection.StaticMappedFolderDetection

    folder = os.path.abspath("C:/Users/rical/OneDrive/Desktop/QSmell_Tool/qsmell-tool/qiskit_algorithms/gradients/reverse")
    output_saving_folder=os.path.abspath("C:/Users/rical/OneDrive/Desktop/SmellResults")




    smells={}

    pyFiles = get_all_python_files(folder)
    for file in pyFiles:
        
        smells[file]=autofix_map_detect(file)

        for smell in smells[file]:
            print(smell.as_dict())

    
        
    save_output(output_saving_folder, smells, folder)
    print("Analysis completed")
"""