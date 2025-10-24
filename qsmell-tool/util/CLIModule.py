import click
import os


def static_method(resource, result_folder=None):

    print(f"🔧 Running STATIC method...")
    print(f"📁 Resource: {resource}")
    
    if result_folder:
        print(f"💾 Results will be saved to: {result_folder}")
    else:
        print(f"💾 Results will be saved to: ./results (default)")
        result_folder = "./results"
    
    # Your actual static method logic here
    # For example:
    # result = perform_static_analysis(resource)
    
    # Mock result for demonstration
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
    
    # Your actual dynamic method logic here
    # For example:
    # result = perform_dynamic_analysis(resource)
    
    # Mock result for demonstration
    result = f"Dynamic analysis completed for {resource}"
    
    return result

@click.command()
@click.option('-static', 'method', flag_value='static', help='Use static analysis method')
@click.option('-dynamic', 'method', flag_value='dynamic', help='Use dynamic analysis method')
@click.argument('resource', type=click.Path(exists=True))
@click.argument('resultfolder', type=click.Path(), required=False, default=None)
def qspire(method, resource, resultfolder):

    try:
        # Check if method was specified
        if method is None:
            click.echo("❌ Error: You must specify either -static or -dynamic", err=True)
            return
        # Validate resource path
        if not os.path.exists(resource):
            click.echo(f"❌ Error: Resource path '{resource}' does not exist!", err=True)
            return
        
        # Execute the appropriate method
        if method == 'static':
            result = static_method(resource, resultfolder)
        elif method == 'dynamic':
            result = dynamic_method(resource, resultfolder)
        
        # Print results
        click.echo("\n" + "="*50)
        click.echo("✅ RESULTS:")
        click.echo("="*50)
        click.echo(result)
        click.echo("="*50 + "\n")
        
    except Exception as e:
        click.echo(f"❌ Error occurred: {str(e)}", err=True)
        raise


if __name__ == "__main__":
    qspire()