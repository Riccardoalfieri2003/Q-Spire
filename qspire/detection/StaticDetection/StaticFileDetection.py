from smells.CG.CG import CG
from smells.IdQ.IdQ import IdQ
from smells.IM.IM import IM
from smells.IQ.IQ import IQ
from smells.LC.LC import LC
from smells.LPQ.LPQ import LPQ
from smells.NC.NC import NC
from smells.ROC.ROC import ROC

from smells.Detector import Detector
from smells.CG.CGDetector import CGDetector
from smells.IdQ.IdQDetector import IdQDetector
from smells.IM.IMDetector import IMDetector
from smells.IQ.IQDetector import IQDetector
from smells.LC.LCDetector import LCDetector
from smells.LPQ.LPQDetector import LPQDetector
from smells.NC.NCDetector import NCDetector
from smells.ROC.ROCDetector import ROCDetector

import importlib
import traceback
import os
import ast
import threading
import builtins

smell_classes = [CG, IdQ, IM, IQ, LC, LPQ, NC, ROC]

thread_local = threading.local()

suppress_print=True

def suppressed_print(*args, **kwargs):
    # Check if current thread should suppress prints
    if getattr(thread_local, 'suppress_prints', False):
        return
    # Otherwise, use original print
    original_print(*args, **kwargs)

# Store original print function
if suppress_print: original_print = builtins.print

# Global set to track files currently being processed (with lock for thread safety)
processing_files = set()
processing_lock = threading.Lock()

# Global counter to track depth of detect_smells_from_file calls
call_depth = 0
call_depth_lock = threading.Lock()

# Configuration for maximum allowed exec depth
MAX_EXEC_DEPTH = 3  # You can change this value

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

class ExecDepthTracker:
    """Thread-safe tracker for exec call depth per thread."""
    
    def __init__(self, max_depth=MAX_EXEC_DEPTH):
        self.max_depth = max_depth
        self.thread_data = {}
        self.lock = threading.Lock()
    
    def increment_depth(self, thread_id):
        """Increment exec depth for a thread and return current depth."""
        with self.lock:
            if thread_id not in self.thread_data:
                self.thread_data[thread_id] = 0
            self.thread_data[thread_id] += 1
            return self.thread_data[thread_id]
    
    def decrement_depth(self, thread_id):
        """Decrement exec depth for a thread."""
        with self.lock:
            if thread_id in self.thread_data:
                self.thread_data[thread_id] = max(0, self.thread_data[thread_id] - 1)
                if self.thread_data[thread_id] == 0:
                    del self.thread_data[thread_id]
    
    def get_depth(self, thread_id):
        """Get current exec depth for a thread."""
        with self.lock:
            return self.thread_data.get(thread_id, 0)
    
    def should_skip(self, thread_id):
        """Check if current depth exceeds maximum allowed depth."""
        return self.get_depth(thread_id) >= self.max_depth

# Global exec depth tracker
exec_tracker = ExecDepthTracker(MAX_EXEC_DEPTH)

def detect_smells_from_file(file: str, max_exec_depth: int = MAX_EXEC_DEPTH):
    """
    Detect smells from a file with exec depth tracking.
    
    Args:
        file: Path to the Python file to analyze
        max_exec_depth: Maximum allowed depth of exec calls (default: MAX_EXEC_DEPTH)
    """
    global call_depth
    
    # Update the global tracker's max depth if specified
    global exec_tracker
    exec_tracker = ExecDepthTracker(max_exec_depth)
    
    # Increment call depth
    with call_depth_lock:
        call_depth += 1
        current_depth = call_depth
    
    try:
        # Normalize the file path to handle different path formats
        normalized_file = os.path.normpath(os.path.abspath(file))
        
        # If this is not the first call (depth > 1), it means a file is calling other files
        if current_depth > 3:
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
            # print(f"No static exec detected in {file}, proceeding with smell detection")
            
            # Set up runtime exec detection with depth tracking
            exec_detected = threading.Event()  # Thread-safe flag
            original_exec = builtins.exec
            original_exec_module = None
            
            # Try to get exec_module from importlib if available
            try:
                import importlib._bootstrap
                original_exec_module = importlib._bootstrap.ModuleSpec.exec_module
            except (ImportError, AttributeError):
                pass
            
            def exec_hook(code, globals_dict=None, locals_dict=None):
                """Hook to detect exec calls and track their depth."""
                thread_id = threading.get_ident()
                
                # Get the call stack to understand where this exec is coming from
                stack = traceback.extract_stack()
                
                # Skip if this exec call is from Python internals that we should ignore
                for frame in stack:
                    filename = frame.filename.lower()
                    function_name = frame.name
                    
                    # Skip exec calls from Python's internal systems (imports, etc.)
                    internal_patterns = [
                        'importlib', 'runpy', 'pkgutil', 'site-packages',
                        'ast.py', 'compile.py', 'types.py', '_bootstrap',
                        'loader.py', 'machinery.py', 'frozen', '<frozen',
                        'zipimport.py', '_bootstrap_external.py'
                    ]
                    
                    internal_functions = [
                        'exec_module', '_load_module_shim', 'load_module',
                        'run_code', 'run_module', '_run_code', '_run_module_as_main',
                        '_compile_bytecode', '_load_unlocked', 'get_code'
                    ]
                    
                    if (any(pattern in filename for pattern in internal_patterns) or 
                        function_name in internal_functions):
                        # This is a legitimate system exec call, don't track depth
                        return original_exec(code, globals_dict, locals_dict)
                
                # This appears to be a user-level exec call, track its depth
                current_exec_depth = exec_tracker.increment_depth(thread_id)
                
                try:
                    # Check if we should skip due to exec depth
                    if current_exec_depth > max_exec_depth:
                        print(f"Detected exec at depth {current_exec_depth} (max: {max_exec_depth}) - marking for skip")
                        exec_detected.set()
                        raise RuntimeError("EXEC_DETECTED_DURING_ANALYSIS")
                    
                    #print(f"Allowing exec at depth {current_exec_depth}")
                    return original_exec(code, globals_dict, locals_dict)
                    
                finally:
                    # Always decrement depth when exec finishes
                    exec_tracker.decrement_depth(thread_id)
            
            def exec_module_hook(self, module):
                """Hook to detect exec_module calls and track their depth."""
                thread_id = threading.get_ident()
                
                # Get the call stack
                stack = traceback.extract_stack()
                
                # Skip if this is from Python internals
                for frame in stack:
                    filename = frame.filename.lower()
                    if any(pattern in filename for pattern in [
                        'importlib', '_bootstrap', 'loader.py', 'machinery.py',
                        'frozen', '<frozen', 'zipimport.py'
                    ]):
                        # This is internal module loading, allow it
                        return original_exec_module(self, module)
                
                # This appears to be a user-level module execution, track depth
                current_exec_depth = exec_tracker.increment_depth(thread_id)
                
                try:
                    if current_exec_depth > max_exec_depth:
                        print(f"Detected exec_module at depth {current_exec_depth} (max: {max_exec_depth}) - marking for skip")
                        exec_detected.set()
                        raise RuntimeError("EXEC_DETECTED_DURING_ANALYSIS")
                    
                    #print(f"Allowing exec_module at depth {current_exec_depth}")
                    return original_exec_module(self, module)
                    
                finally:
                    exec_tracker.decrement_depth(thread_id)
            
            # Import the detector classes (assuming they're available)
            try:
                #smell_classes = [CG, IdQ, IM, IQ, LC, LPQ, NC, ROC]
                detector_objects = [Detector(smell_cls) for smell_cls in smell_classes]
            except NameError:
                print("Detector classes not imported. Please import CG, IdQ, IM, IQ, LC, LPQ, NC, ROC, Detector")
                return []

            smells_lock = threading.Lock()
            all_smells = []
            
            # Replace print globally with our controlled version
            original_print_backup = builtins.print
            if suppress_print: builtins.print = suppressed_print
            
            # Replace exec globally with our hook
            builtins.exec = exec_hook
            
            # Replace exec_module if available
            if original_exec_module:
                try:
                    import importlib._bootstrap
                    importlib._bootstrap.ModuleSpec.exec_module = exec_module_hook
                except (ImportError, AttributeError):
                    pass

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
                    print(f"Skipping {file} - exec() was called during analysis at depth > {max_exec_depth}")
                    return []

                return all_smells
                
            finally:
                # Always restore original functions
                builtins.print = original_print_backup
                builtins.exec = original_exec
                
                # Restore exec_module if we modified it
                if original_exec_module:
                    try:
                        import importlib._bootstrap
                        importlib._bootstrap.ModuleSpec.exec_module = original_exec_module
                    except (ImportError, AttributeError):
                        pass
        
        finally:
            # Always remove file from processing set when done
            with processing_lock:
                processing_files.discard(normalized_file)
    
    finally:
        # Decrement call depth
        with call_depth_lock:
            call_depth -= 1






if __name__ == "__main__":
    # python -m test.StaticFileDetection
    
    file = os.path.abspath("C:/Users/rical/OneDrive/Desktop/QSmell_Tool/qspire/generated_executables/executable_initializer.py")
    if os.path.exists(file):
        # You can change the max_exec_depth here (default is 3)
        result = detect_smells_from_file(file, max_exec_depth=3)
        if result is not None:
            for smell in result: 
                print(smell.as_dict())
        print(result)
    else:
        print(f"File {file} not found")
