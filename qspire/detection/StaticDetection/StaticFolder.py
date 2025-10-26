import csv
from pathlib import Path
import shutil
import threading
"""from test.StaticCircuit import FunctionExecutionGenerator
from test.AutoFix import auto_fix
import os
from test.GeneralFileTest import detect_smells_from_file
from test.GeneralFolderTest import save_output"""

from detection.StaticDetection.StaticCircuit import FunctionExecutionGenerator
from detection.StaticDetection.AutoFix import auto_fix
import os
from detection.DynamicDetection.GeneralFileTest import detect_smells_from_file
from detection.DynamicDetection.GeneralFolderTest import save_output


def get_all_python_files(folder_path):
    """Get all Python files in a folder recursively."""
    python_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                python_files.append(full_path)
    return python_files

def clear_folder(folder_path):
    folder_path = os.path.abspath(folder_path)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Recursively delete a folder
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")




if __name__ == "__main__":
    # python -m test.StaticFolder
    # python -m detection.StaticDetection.StaticFolder

    folder = os.path.abspath("C:/Users/rical/OneDrive/Desktop/Qelm-main")
    output_saving_folder=os.path.abspath("C:/Users/rical/OneDrive/Desktop/SmellResults")



    pyFiles = get_all_python_files(folder)

    threads = []
    executables_dict = {}  # { executable_path: source_file_path }

    generator = FunctionExecutionGenerator()

    for file in pyFiles:
        #print(f"Analyzing {file}")
        try: 
            functions_with_circuits = generator.find_all_functions_with_circuits(open(file, 'r', encoding="utf-8").read(), file)

            output_directory = "generated_executables"
            executables = generator.analyze_and_generate_all_executables(file, output_directory)
            print(f"\nGenerated {len(executables)} executable files in '{output_directory}/' directory for file {file}")

            # Map each generated executable to its original file
            for exe in executables:
                # Compose full filename: executable_<function_name>.py
                exe_filename = f"executable_{exe}.py"
                
                # Join with output directory to get full path
                abs_exe_path = os.path.abspath(os.path.join(output_directory, exe_filename))
                abs_source_path = os.path.abspath(file)

                executables_dict[abs_exe_path] = abs_source_path

        
            
            for exe in executables_dict:
                t = threading.Thread(target=auto_fix, args=(Path(exe),))
                t.start()
                threads.append(t)

            # Wait for all threads to complete
            for t in threads:
                t.join()
            



            
            for exe in executables_dict:
                try:
                    smells = detect_smells_from_file(exe)
                    if len(smells) > 0:

                        for smell in smells:
                            print(smell.as_dict())

                        source_file = executables_dict[exe]

                        # Step 1: Get subfolder path (relative to the analyzed folder)
                        relative_source_path = os.path.relpath(source_file, folder)  # folder is the root you're analyzing
                        save_subfolder = os.path.join(output_saving_folder, relative_source_path)

                        # Step 2: Ensure the folder exists
                        os.makedirs(save_subfolder, exist_ok=True)

                        # Step 3: Clean the filename: remove "executable_" and ".py"
                        exe_basename = os.path.basename(exe)  # e.g., executable___init__.py
                        function_name = os.path.splitext(exe_basename)[0].replace("executable_", "")  # __init__

                        # Step 4: Build path for output CSV
                        output_csv_path = os.path.join(save_subfolder, f"{function_name}.csv")

                        # Step 5: Write smells to CSV
                        dict_rows = [smell.as_dict() for smell in smells]
                        if dict_rows:
                            fieldnames = sorted({k for d in dict_rows for k in d.keys()})

                            with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
                                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(dict_rows)

                            print(f"Saved {len(dict_rows)} smells to {output_csv_path}")


                except Exception as e:
                    print(e)





            #CLEAR SECTION

            for ex in executables_dict:
                folder_path = os.path.dirname(ex)
                clear_folder(folder_path)
                break
    
            executables_dict.clear()
            

            
            
        except Exception as e:
            print(f"Error analyzing {file}: {e}")
            continue

    print("Analysis completed")