import click
import os
import sys

from detection.DynamicDetection.GeneralFileTest import detect_smells_from_file


def static_method(resource, result_folder=None):
    print(f"🔧 Running STATIC method...")
    print(f"📁 Resource: {resource}")
    
    if result_folder:
        print(f"💾 Results will be saved to: {result_folder}")
    else:
        print(f"💾 Results will be saved to: ./results (default)")
        result_folder = "./results"
    
    result = f"Static analysis completed for {resource}"
    return result





def dynamic_method(resource, result_folder=None):
    print(f"🔧 Running DYNAMIC method...")
    print(f"📁 Resource: {resource}")


    
    if result_folder:
        print(f"💾 Results will be saved to: {result_folder}")
    else:
        print(f"💾 Results will be saved to: ./results (default)")
        result_folder = "./results"


    result=detect_smells_from_file(resource)
    for smell in result:
        print(smell.as_dict())

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
            #raise click.Abort()
        
        # NOW check if resource exists
        if not os.path.exists(resource):
            click.echo(f"❌ Error: Resource path '{resource}' does not exist!", err=True)
            sys.exit(1)
            #raise click.Abort()
        
        # Execute the appropriate method
        if method == 'static': result = static_method(resource, outputfolder)
        elif method == 'dynamic': result = dynamic_method(resource, outputfolder)
        
        # Print results
        click.echo("\n" + "="*50)
        click.echo("✅ RESULTS:")
        click.echo(result)
        click.echo("="*50 + "\n")
        
    except click.Abort:
        # Clean exit for user errors
        sys.exit(1)
        raise
    except Exception as e:
        click.echo(f"❌ Error occurred: {str(e)}", err=True)
        sys.exit(1)
        raise


if __name__ == "__main__":
    qspire()


