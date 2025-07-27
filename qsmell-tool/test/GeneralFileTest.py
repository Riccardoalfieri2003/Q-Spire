import threading
import io
from contextlib import redirect_stdout, redirect_stderr
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

import threading
import builtins
import os
import threading
import builtins
import os

import os
import ast
import threading
import builtins

"""
# Thread-local storage for print suppression
thread_local = threading.local()

def suppressed_print(*args, **kwargs):
    # Check if current thread should suppress prints
    if getattr(thread_local, 'suppress_prints', False):
        return
    # Otherwise, use original print
    original_print(*args, **kwargs)

# Store original print function
original_print = builtins.print

# Global set to track files currently being processed (with lock for thread safety)
processing_files = set()
processing_lock = threading.Lock()

# Global counter to track depth of detect_smells_from_file calls
call_depth = 0
call_depth_lock = threading.Lock()

def has_local_imports(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Get the directory of the current file to check for local modules
        file_dir = os.path.dirname(os.path.abspath(file_path))
        
        # Check all import statements
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if is_local_module(alias.name, file_dir):
                        print(f"File {file_path} imports local module: {alias.name}")
                        return True
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    if is_local_module(node.module, file_dir):
                        print(f"File {file_path} imports from local module: {node.module}")
                        return True
                else:
                    # Relative imports (from . import ...)
                    print(f"File {file_path} uses relative imports")
                    return True
        
        return False
        
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError) as e:
        print(f"Error analyzing {file_path}: {e}")
        # If we can't parse the file, assume it might have local imports (be safe)
        return True

def is_local_module(module_name, file_dir):
    if not module_name:
        return False
    
    # Skip well-known standard library and common third-party modules
    stdlib_modules = {
        'os', 'sys', 'threading', 'builtins', 'ast', 'subprocess', 'tempfile',
        'pathlib', 'json', 'time', 'datetime', 'collections', 'itertools',
        'functools', 're', 'math', 'random', 'uuid', 'hashlib', 'urllib',
        'http', 'email', 'xml', 'html', 'sqlite3', 'csv', 'pickle', 'logging',
        'argparse', 'configparser', 'glob', 'shutil', 'zipfile', 'tarfile',
        'gzip', 'bz2', 'lzma', 'base64', 'binascii', 'struct', 'codecs',
        'locale', 'gettext', 'string', 'textwrap', 'unicodedata', 'stringprep',
        'readline', 'rlcompleter', 'cmd', 'shlex', 'typing', 'types',
        'copy', 'pprint', 'reprlib', 'enum', 'numbers', 'cmath', 'decimal',
        'fractions', 'statistics', 'array', 'weakref', 'gc', 'inspect',
        'site', 'importlib', 'keyword', 'pkgutil', 'modulefinder', 'runpy',
        'traceback', 'future', 'warnings', 'contextlib', 'abc', 'atexit',
        'tracemalloc', 'linecache', 'token', 'tokenize', 'dis', 'py_compile',
        'compileall', 'zipapp', 'venv', 'ensurepip', 'zipimport', 'pkgutil',
        # Common third-party libraries
        'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn', 'tensorflow',
        'torch', 'requests', 'flask', 'django', 'pytest', 'unittest',
        'click', 'jinja2', 'werkzeug', 'sqlalchemy', 'alembic', 'celery',
        'redis', 'pymongo', 'psycopg2', 'mysql', 'boto3', 'awscli',
        'docker', 'kubernetes', 'yaml', 'toml', 'lxml', 'beautifulsoup4',
        'scrapy', 'selenium', 'pillow', 'opencv', 'imageio', 'networkx',
        'plotly', 'seaborn', 'statsmodels', 'sympy', 'nltk', 'spacy',
        'transformers', 'huggingface', 'openai', 'anthropic'
    }
    
    base_module = module_name.split('.')[0]
    if base_module in stdlib_modules:
        return False
    
    # Check if it's a relative import (starts with .)
    if module_name.startswith('.'):
        return True
    
    # Check if there's a corresponding Python file in the same directory
    potential_paths = [
        os.path.join(file_dir, base_module + '.py'),
        os.path.join(file_dir, base_module, '__init__.py'),
    ]
    
    # Also check parent and sibling directories (common project structures)
    parent_dir = os.path.dirname(file_dir)
    if parent_dir != file_dir:  # Avoid infinite loop
        potential_paths.extend([
            os.path.join(parent_dir, base_module + '.py'),
            os.path.join(parent_dir, base_module, '__init__.py'),
        ])
        
        # Check sibling directories
        if os.path.exists(parent_dir):
            for item in os.listdir(parent_dir):
                sibling_path = os.path.join(parent_dir, item)
                if os.path.isdir(sibling_path) and item != os.path.basename(file_dir):
                    potential_paths.extend([
                        os.path.join(sibling_path, base_module + '.py'),
                        os.path.join(sibling_path, base_module, '__init__.py'),
                    ])
    
    # Check if any of these paths exist
    for path in potential_paths:
        if os.path.exists(path):
            return True
    
    return False

def detect_smells_from_file(file: str):
    global call_depth
    
    # Increment call depth
    with call_depth_lock:
        call_depth += 1
        current_depth = call_depth
    
    try:
        # Normalize the file path to handle different path formats
        normalized_file = os.path.normpath(os.path.abspath(file))
        
        # Check if this file imports other local Python files
        if has_local_imports(normalized_file):
            print(f"Skipping {file} - file imports/calls other local Python files")
            return []
        
        # If this is not the first call (depth > 1), it means a file is calling other files
        if current_depth > 1:
            print(f"Skipping {file} - called from another file being analyzed")
            return []
        
        # Check if this file is already being processed (direct recursion detection)
        with processing_lock:
            if normalized_file in processing_files:
                print(f"Skipping {file} - direct recursion detected")
                return []
            
            # Add file to processing set
            processing_files.add(normalized_file)
        
        try:
            smell_classes = [CG, IdQ, IM, IQ, LC, LPQ, NC, ROC]
            detector_objects = [Detector(smell_cls) for smell_cls in smell_classes]

            smells_lock = threading.Lock()
            all_smells = []
            
            # Replace print globally with our controlled version
            builtins.print = suppressed_print

            def run_detection(detector):
                # Set thread-local flag to suppress prints in this thread
                thread_local.suppress_prints = True
                
                try:
                    smells = detector.detect(file)
                    with smells_lock:
                        all_smells.extend(smells)
                except Exception:
                    pass
                finally:
                    # Clear the suppression flag for this thread
                    thread_local.suppress_prints = False

            threads = []
            for detector in detector_objects:
                thread = threading.Thread(target=run_detection, args=(detector,))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

            # Restore original print function
            builtins.print = original_print

            return all_smells
        
        finally:
            # Always remove file from processing set when done
            with processing_lock:
                processing_files.discard(normalized_file)
    
    finally:
        # Decrement call depth
        with call_depth_lock:
            call_depth -= 1


if __name__ == "__main__":
    # Test with the problematic file
    file = os.path.abspath("C:/Users/rical/OneDrive/Desktop/QSmell_Tool/qsmell-tool/test/LC/LCCode.py")
    result = detect_smells_from_file(file)
    for smell in result:
        print(smell.as_dict())
    
"""



import os
import ast
import threading
import builtins
import sys
import importlib.util
from types import ModuleType










































thread_local = threading.local()

def suppressed_print(*args, **kwargs):
    # Check if current thread should suppress prints
    if getattr(thread_local, 'suppress_prints', False):
        return
    # Otherwise, use original print
    original_print(*args, **kwargs)

# Store original print function
original_print = builtins.print

# Global set to track files currently being processed (with lock for thread safety)
processing_files = set()
processing_lock = threading.Lock()

# Global counter to track depth of detect_smells_from_file calls
call_depth = 0
call_depth_lock = threading.Lock()

def contains_exec_comprehensive(file_path):
    """
    Comprehensive check for exec() usage in a Python file and its dependencies.
    This checks both static analysis and import analysis.
    """
    checked_files = set()
    
    def check_file_for_exec(path):
        """Recursively check a file and its imports for exec usage."""
        if path in checked_files:
            return False
        
        checked_files.add(path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (FileNotFoundError, UnicodeDecodeError) as e:
            print(f"Could not read {path}: {e}")
            return True  # Assume it contains exec if we can't read it
        
        # Quick string check first
        if 'exec(' in content or 'exec ' in content:
            print(f"File {path} contains 'exec' in source code")
            return True
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"Syntax error in {path}: {e}")
            return True  # Assume it contains exec if we can't parse it
        
        # Check for exec() function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Direct exec() call
                if isinstance(node.func, ast.Name) and node.func.id == 'exec':
                    print(f"File {path} contains exec() function call")
                    return True
                # builtins.exec or __builtins__.exec
                elif isinstance(node.func, ast.Attribute):
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id in ['builtins', '__builtins__'] and 
                        node.func.attr == 'exec'):
                        print(f"File {path} contains {node.func.value.id}.exec() function call")
                        return True
        
        # Check imports for potential exec usage
        file_dir = os.path.dirname(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if check_import_for_exec(alias.name, file_dir):
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and check_import_for_exec(node.module, file_dir):
                    return True
        
        return False
    
    def check_import_for_exec(module_name, base_dir):
        """Check if an imported module contains exec."""
        if not module_name:
            return False
        
        # Skip standard library and common packages that we know don't use exec
        skip_modules = {
            'qiskit', 'numpy', 'scipy', 'matplotlib', 'pandas', 'os', 'sys', 
            'json', 'csv', 'math', 'random', 'datetime', 're', 'collections',
            'itertools', 'functools', 'operator', 'pathlib', 'typing',
            'gettext', 'locale', 'threading', 'queue', 'urllib', 'http',
            'socket', 'ssl', 'email', 'html', 'xml', 'logging', 'unittest',
            'importlib', 'pkgutil', 'warnings', 'weakref', 'gc', 'copy',
            'pickle', 'struct', 'zlib', 'gzip', 'bz2', 'lzma', 'tarfile',
            'zipfile', 'hashlib', 'hmac', 'secrets', 'uuid', 'time',
            'calendar', 'argparse', 'shlex', 'glob', 'fnmatch', 'linecache',
            'shutil', 'stat', 'filecmp', 'tempfile', 'contextlib', 'abc',
            'numbers', 'cmath', 'decimal', 'fractions', 'statistics',
            'array', 'bisect', 'heapq', 'copy', 'pprint', 'reprlib',
            'enum', 'graphlib', 'string', 'textwrap', 'unicodedata',
            'stringprep', 'readline', 'rlcompleter', 'io', 'codecs'
        }
        
        root_module = module_name.split('.')[0]
        if root_module in skip_modules:
            return False
        
        # Skip if it's in the Python standard library path
        try:
            spec = importlib.util.find_spec(module_name)
            if spec and spec.origin:
                # Check if it's in the standard library
                import sysconfig
                stdlib_path = sysconfig.get_path('stdlib')
                if spec.origin.startswith(stdlib_path):
                    return False
                
                # Check if it's in site-packages (third-party)
                if 'site-packages' in spec.origin:
                    # Only check local modules in our project directory
                    return False
        except (ImportError, ModuleNotFoundError, ValueError):
            pass
        
        # Only check local project files
        try:
            # First, try relative import (local project files)
            relative_path = os.path.join(base_dir, module_name.replace('.', os.sep) + '.py')
            if os.path.exists(relative_path):
                return check_file_for_exec(relative_path)
            
            # Check for package-style imports in the same directory
            package_path = os.path.join(base_dir, module_name.replace('.', os.sep), '__init__.py')
            if os.path.exists(package_path):
                return check_file_for_exec(package_path)
                
        except Exception:
            pass
        
        return False
    
    return check_file_for_exec(file_path)

def has_exec_function(file_path):
    """
    Check if a Python file uses the exec() function in any form.
    This is the simplified version for basic checking.
    """
    return contains_exec_comprehensive(file_path)

def detect_smells_from_file(file: str):
    global call_depth
    
    # Increment call depth
    with call_depth_lock:
        call_depth += 1
        current_depth = call_depth
    
    try:
        # Normalize the file path to handle different path formats
        normalized_file = os.path.normpath(os.path.abspath(file))
        
        # If this is not the first call (depth > 1), it means a file is calling other files
        if current_depth > 1:
            print(f"Skipping {file} - called from another file being analyzed")
            return []
        
        # Check if this file is already being processed (direct recursion detection)
        with processing_lock:
            if normalized_file in processing_files:
                print(f"Skipping {file} - direct recursion detected")
                return []
            
            # Add file to processing set
            processing_files.add(normalized_file)
        
        try:
            # First, do static analysis to check for exec in the file itself
            if has_exec_function(file):
                print(f"Skipping {file} - contains exec() function")
                return []
            
            # Proceed with detection but monitor for exec calls during runtime
            print(f"No static exec detected in {file}, proceeding with smell detection")
            
            # Set up runtime exec detection
            exec_detected = threading.Event()  # Thread-safe flag
            original_exec = builtins.exec
            
            def exec_hook(code, globals_dict=None, locals_dict=None):
                """Hook to detect problematic exec calls during detection."""
                # Get the call stack to understand where this exec is coming from
                import traceback
                import inspect
                
                # Get the current stack
                stack = traceback.extract_stack()
                
                # Skip if this exec call is from Python internals or common analysis tools
                for frame in stack:
                    filename = frame.filename.lower()
                    function_name = frame.name
                    
                    # Skip exec calls from Python's internal systems
                    if (any(internal in filename for internal in [
                        'importlib', 'runpy', 'pkgutil', 'site-packages',
                        'ast.py', 'compile.py', 'types.py', '_bootstrap',
                        'loader.py', 'machinery.py'
                    ]) or function_name in [
                        'exec_module', '_load_module_shim', 'load_module',
                        'run_code', 'run_module', '_run_code', '_run_module_as_main'
                    ]):
                        # This is a legitimate system exec call
                        return original_exec(code, globals_dict, locals_dict)
                
                # Now check if this exec is happening as part of executing the target file's code
                target_file_normalized = os.path.normpath(os.path.abspath(file))
                
                # Look for signs that this exec is executing code from our target file
                code_str = str(code) if hasattr(code, '__str__') else repr(code)
                
                # Check the globals dictionary for clues about what's being executed
                if globals_dict:
                    file_from_globals = globals_dict.get('__file__', '')
                    if file_from_globals:
                        file_from_globals_normalized = os.path.normpath(os.path.abspath(file_from_globals))
                        if file_from_globals_normalized == target_file_normalized:
                            # This exec is executing code from our target file!
                            print(f"Detected exec of target file code: {file_from_globals}")
                            exec_detected.set()
                            raise RuntimeError("EXEC_DETECTED_DURING_ANALYSIS")
                
                # Check if the code being exec'd contains content that looks like it's from our file
                if isinstance(code, str) and len(code) > 50:  # Only check substantial code blocks
                    # Read the target file to compare
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            target_content = f.read()
                        
                        # Check if the exec'd code contains significant portions of our target file
                        lines_from_code = [line.strip() for line in code_str.split('\n') if line.strip() and not line.strip().startswith('#')]
                        lines_from_target = [line.strip() for line in target_content.split('\n') if line.strip() and not line.strip().startswith('#')]
                        
                        if len(lines_from_code) > 5:  # Only check substantial code blocks
                            matches = sum(1 for line in lines_from_code if line in lines_from_target)
                            match_ratio = matches / len(lines_from_code)
                            
                            if match_ratio > 0.3:  # If more than 30% of exec'd code matches our target file
                                print(f"Detected exec of code similar to target file (match ratio: {match_ratio:.2f})")
                                exec_detected.set()
                                raise RuntimeError("EXEC_DETECTED_DURING_ANALYSIS")
                    
                    except (FileNotFoundError, UnicodeDecodeError, OSError):
                        pass  # Can't read target file, continue with exec
                
                # If we get here, it's probably a legitimate exec call not related to our target
                return original_exec(code, globals_dict, locals_dict)
            
            # Import the detector classes (assuming they're available)
            try:
                smell_classes = [CG, IdQ, IM, IQ, LC, LPQ, NC, ROC]
                detector_objects = [Detector(smell_cls) for smell_cls in smell_classes]
            except NameError:
                print("Detector classes not imported. Please import CG, IdQ, IM, IQ, LC, LPQ, NC, ROC, Detector")
                return []

            smells_lock = threading.Lock()
            all_smells = []
            
            # Replace print globally with our controlled version
            original_print_backup = builtins.print
            builtins.print = suppressed_print
            
            # Replace exec globally with our hook
            builtins.exec = exec_hook

            def run_detection(detector):
                # Set thread-local flag to suppress prints in this thread
                thread_local.suppress_prints = True
                
                try:
                    smells = detector.detect(file)
                    with smells_lock:
                        all_smells.extend(smells)
                except RuntimeError as e:
                    if "EXEC_DETECTED_DURING_ANALYSIS" in str(e):
                        # This is our exec detection - don't add to results
                        pass
                    else:
                        # Some other runtime error, log it
                        print(f"Runtime error in detector {detector}: {e}")
                except Exception as e:
                    # Any other exception, log it but don't crash
                    print(f"Error in detector {detector}: {e}")
                finally:
                    # Clear the suppression flag for this thread
                    thread_local.suppress_prints = False

            try:
                threads = []
                for detector in detector_objects:
                    thread = threading.Thread(target=run_detection, args=(detector,))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                # Check if exec was detected during the analysis
                if exec_detected.is_set():
                    print(f"Skipping {file} - exec() was called during analysis")
                    return []

                return all_smells
                
            finally:
                # Always restore original functions
                builtins.print = original_print_backup
                builtins.exec = original_exec
        
        finally:
            # Always remove file from processing set when done
            with processing_lock:
                processing_files.discard(normalized_file)
    
    finally:
        # Decrement call depth
        with call_depth_lock:
            call_depth -= 1






































if __name__ == "__main__":

    """python -m test.GeneralFileTest"""

    # Test the exec detection
    
    # Test with the problematic file
    file = os.path.abspath("C:/Users/rical/OneDrive/Desktop/QSmell_Tool/qsmell-tool/test/LC/LCCode.py")
    if os.path.exists(file):
        result = detect_smells_from_file(file)
        if result is not None:
            for smell in result: print(smell.as_dict())
        print(f"Result for {file}: {result}")
    else:
        print(f"File {file} not found")