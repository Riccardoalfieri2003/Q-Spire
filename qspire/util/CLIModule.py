import csv
import click
import os
import sys

from detection.DynamicDetection.GeneralFileTest import dynamic_file_detect, dynamic_folder_detect
from detection.StaticDetection.StaticMappedFolderDetection import static_file_detect, static_folder_detect



def is_file(resource:str):
    return True if resource.endswith(".py") else False

def save_output_for_folders(output_saving_folder, smells, folder="SmellResults"):
    os.makedirs(output_saving_folder, exist_ok=True)

    # Create a subfolder
    subfolder_path = os.path.join(output_saving_folder, folder)
    os.makedirs(subfolder_path, exist_ok=True)

    for file_path, smell_list in smells.items():
        # Skip if no smells
        if not smell_list:
            continue

        # Encode full path into a valid filename
        safe_name = file_path.replace(folder,"").replace(":", "_").replace("\\", "_").replace("/", "_")
        output_file_path = os.path.join(subfolder_path, f"{safe_name}.csv")

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

    print("Saved")



def save_output_for_files(resource, output_saving_folder, smells, folder="SmellResults"):
    os.makedirs(output_saving_folder, exist_ok=True)


    if len(smells)==0:
        print("No Smells Found")
        return

    # Encode full path into a valid filename
    safe_name = resource.replace(folder,"").replace(":", "_").replace("\\", "_").replace("/", "_")
    output_file_path = os.path.join(output_saving_folder, f"{safe_name}.csv")

    # Convert all smells to dict form
    smells_dicts = [s.as_dict() for s in smells]

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

    print("Results Saved")






def static_method(resource, result_folder=None):
    print(f"🔧 Running STATIC method...")
    print(f"📁 Resource: {resource}")

    if is_file(resource):
        result=static_file_detect(resource)

        if result_folder: 
            subfolder=resource.split("\\")[-1].replace(".py","")
            print(f"💾 Results will be saved to: {result_folder}\{subfolder}")
            save_output_for_files(resource, result_folder, result, subfolder )
        else: print(f"Results will be shown on the terminal (default)")
    

    else:
        result=static_folder_detect(resource)

        if result_folder: 
            subfolder=resource.split("\\")[-1].replace(".py","")
            print(f"💾 Results will be saved to: {result_folder}\{subfolder}")
            save_output_for_folders(result_folder, result, subfolder)
        else: print(f"Results will be shown on the terminal (default)")
    
    
    return result




def dynamic_method(resource, result_folder=None):
    print(f"🔧 Running DYNAMIC method...")
    print(f"📁 Resource: {resource}")

    if is_file(resource):
        result=dynamic_file_detect(resource)

        if result_folder: 
            subfolder=resource.split("\\")[-1].replace(".py","")
            print(f"💾 Results will be saved to: {result_folder}\{subfolder}")
            save_output_for_files(resource, result_folder, result, subfolder )
        else: print(f"Results will be shown on the terminal (default)")
    

    else:
        result=dynamic_folder_detect(resource)

        if result_folder: 
            subfolder=resource.split("\\")[-1].replace(".py","")
            print(f"💾 Results will be saved to: {result_folder}\{subfolder}")
            save_output_for_folders(result_folder, result, subfolder)
        else: print(f"Results will be shown on the terminal (default)")
    
    
    return result





@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-static', 'method', flag_value='static', help='Use static analysis method')
@click.option('-dynamic', 'method', flag_value='dynamic', help='Use dynamic analysis method')
@click.argument('resource', type=click.Path(), required=True)
@click.argument('outputfolder', type=click.Path(), required=False, default=None)
def qspire(method, resource, outputfolder):
    """
    QSpire - Quantum Code Analysis Tool
    
    Analyzes quantum code for Code Smells.
    
    \b
    Arguments:
      resource       Path to the file or directory to analyze (required)
      outputfolder   Path where results will be saved (optional, if not expliceted prints results in the terminal)
    
    \b
    Examples:
      qspire -static "myfile.py"
      qspire -dynamic "myfile.py" "../output"
    """
    
    try:
        # Check if method was specified FIRST
        if method is None:
            click.echo("Error: You must specify either -static or -dynamic", err=True)
            click.echo("\nUsage: qspire (-static | -dynamic) resource [outputfolder]")
            click.echo("Try 'qspire --help' for more information.")
            sys.exit(1)
        
        # NOW check if resource exists
        if not os.path.exists(resource):
            click.echo(f"❌ Error: Resource path '{resource}' does not exist!", err=True)
            sys.exit(1)
        
        # Execute the appropriate method
        if method == 'static': result = static_method(resource, outputfolder)
        elif method == 'dynamic': result = dynamic_method(resource, outputfolder)
        
        # Print results
        click.echo("\n" + "="*50)
        click.echo("✅ Results:")

        if is_file(resource): 
            for smell in result: print(smell.as_dict())
        else:
            for file in result:

                if len(result[file])==0: 
                    print(f"No Smells in {file}\n")
                    continue

                print(f"Smells in {file}:")
                for smell in result[file]: print(smell.as_dict())

                print()

        click.echo("="*50 + "\n")
        
    
    except Exception as e:
        click.echo(f"❌ Error occurred: {str(e)}", err=True)
        sys.exit(1)
        raise


if __name__ == "__main__":
    qspire()