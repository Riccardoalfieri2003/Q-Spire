#!/usr/bin/env python3
import subprocess
import sys
import re
import os
from pathlib import Path


def parse_traceback(stderr_text, target_fname):
    """
    Parse the last frame in a Python traceback to extract
    the error type, message, and line number in target_fname.
    """
    tb_lines = stderr_text.strip().splitlines()
    pattern = rf'  File "{re.escape(str(target_fname))}", line (\d+)'
    line_no = None
    for ln in reversed(tb_lines):
        m = re.match(pattern, ln)
        if m:
            line_no = int(m.group(1))
            break
    exc_line = tb_lines[-1]
    exc_match = re.match(r'(\w+(?:Error|Exception)): (.+)', exc_line)
    if not exc_match:
        # Try to match just the error type without message
        exc_match = re.match(r'(\w+(?:Error|Exception))$', exc_line)
        if exc_match:
            return exc_match.group(1), "", line_no
        return None
    err_type, err_msg = exc_match.groups()
    return err_type, err_msg, line_no


def extract_variable_names(line_text):
    """Extract variable names from a line of code for better context."""
    # Find variables being assigned
    assign_vars = re.findall(r'([a-zA-Z_]\w*)\s*=', line_text)
    # Find function calls
    func_calls = re.findall(r'([a-zA-Z_]\w*)\s*\(', line_text)
    # Find attribute access
    attr_access = re.findall(r'([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)', line_text)
    # Find indexing
    indexing = re.findall(r'([a-zA-Z_]\w*)\[', line_text)
    
    return {
        'assigns': assign_vars,
        'calls': func_calls,
        'attributes': attr_access,
        'indexing': indexing
    }

def make_fix(err_type, err_msg, line_text, lines, line_no):
    """
    Return Python code lines to insert before the error line,
    given the error type, message, and the source line text.
    Also receives the full lines array and line number for context analysis.
    """
    indent_match = re.match(r'(\s*)', line_text)
    indent = indent_match.group(1) if indent_match else ''

    print(f"Error type: {err_type}")
    
    # Check if we're in an except block
    def is_in_except_block(lines, current_line_no):
        """Check if the current line is inside an except block"""
        for i in range(current_line_no - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('except ') and line.endswith(':'):
                return True
            elif line.startswith('try:') or (line.startswith('def ') and line.endswith(':')):
                return False
        return False
    
    # Special handling for errors in except blocks
    in_except = is_in_except_block(lines, line_no)
    
    def infer_variable_type(var_name, try_lines):
        """Infer the expected type of a variable based on how it's assigned in try block"""
        for line in try_lines:
            if f'{var_name} =' in line:
                # Look for common patterns to infer type
                assignment = line.split('=', 1)[1].strip()
                
                # Function/method calls that typically return specific types
                if '.result()' in assignment:
                    return 'dict'  # Results are often dict-like
                elif '.run(' in assignment:
                    return 'object'  # Job objects
                elif '.get(' in assignment or '.json(' in assignment:
                    return 'dict'
                elif '.read(' in assignment or '.text' in assignment:
                    return 'str'
                elif '.items(' in assignment or '.keys(' in assignment:
                    return 'list'
                elif 'len(' in assignment or '.count(' in assignment:
                    return 'int'
                elif '[' in assignment and ']' in assignment:
                    return 'list'  # Indexing usually returns from lists
                elif '{' in assignment and '}' in assignment:
                    return 'dict'
                elif '"' in assignment or "'" in assignment:
                    return 'str'
                elif assignment.isdigit() or '.' in assignment and assignment.replace('.', '').isdigit():
                    return 'int' if assignment.isdigit() else 'float'
                elif 'True' in assignment or 'False' in assignment:
                    return 'bool'
                elif 'list(' in assignment or '[' in assignment:
                    return 'list'
                elif 'dict(' in assignment or '{' in assignment:
                    return 'dict'
                else:
                    return 'object'  # Generic object for unknown cases
        return 'object'  # Default fallback
    
    def create_default_value(var_type):
        """Create appropriate default value based on inferred type"""
        defaults = {
            'str': '""',
            'int': '0',
            'float': '0.0',
            'bool': 'False',
            'list': '[]',
            'dict': '{}',
            'object': 'type("MockObject", (), {"__getattr__": lambda self, name: lambda *args, **kwargs: None, "__call__": lambda self, *args, **kwargs: None, "__iter__": lambda self: iter([]), "__len__": lambda self: 0, "__getitem__": lambda self, key: None, "__str__": lambda self: "", "__int__": lambda self: 0})()'
        }
        return defaults.get(var_type, defaults['object'])
    
    # ============ TYPE ERRORS ============
    if err_type == 'TypeError':
        # Handle specific subscriptable errors BEFORE general except block logic
        if 'object is not subscriptable' in err_msg:
            # Extract the object that's not subscriptable
            if '.values[' in line_text:
                # Example: results.values[...]
                obj_match = re.search(r'(\w+)\.values\[', line_text)
                if obj_match:
                    obj_name = obj_match.group(1)
                    # Use a temporary variable to hold values, depending on whether it's a method or attribute
                    return {
                        'action': 'replace',
                        'lines': [
                            f"{indent}# auto-fix: safely extract values from {obj_name}",
                            f"{indent}if hasattr({obj_name}, 'values'):",
                            f"{indent}    _vals = {obj_name}.values() if callable({obj_name}.values) else {obj_name}.values",
                            f"{indent}else:",
                            f"{indent}    _vals = []  # fallback in case values is missing",
                            f"{indent}result = list(_vals)[partial_sum_n : partial_sum_n + n]",
                        ]
                    }

            
            # General case for other subscriptable errors
            subscriptable_match = re.search(r"'([^']+)' object is not subscriptable", err_msg)
            if subscriptable_match:
                obj_type = subscriptable_match.group(1)
                # Find the variable being subscripted
                bracket_match = re.search(r'([a-zA-Z_][\w\.]*)\[', line_text)
                if bracket_match:
                    var_name = bracket_match.group(1)
                    return {'action': 'insert', 'lines': [
                        f"{indent}# auto-fix: make {var_name} subscriptable",
                        f"{indent}if not hasattr({var_name}, '__getitem__'):",
                        f"{indent}    {var_name} = [] if {var_name} is None else list({var_name}) if hasattr({var_name}, '__iter__') else []{var_name}]",
                    ]}
                
        elif "'function' object is not iterable" in err_msg:
            # Detect function call: e.g. key = some_func(arg)
            call_match = re.search(r'(\w+)\s*=\s*(\w+)\s*\(', line_text)
            if call_match:
                assigned_var, func_name = call_match.groups()

                # Try to infer return type by scanning previous lines
                func_def_pattern = re.compile(rf'\s*def {re.escape(func_name)}\s*\(.*?\):')
                return_type = '1'  # Default fallback

                for prev_line in reversed(lines[:line_no]):
                    if func_def_pattern.match(prev_line):
                        # Found the function definition
                        j = lines.index(prev_line) + 1
                        while j < len(lines):
                            inner_line = lines[j].strip()
                            if inner_line.startswith("return"):
                                return_expr = inner_line.split("return", 1)[1].strip()
                                if return_expr.startswith('['):
                                    return_type = '[]'
                                elif return_expr.startswith('{'):
                                    return_type = '{}'
                                elif return_expr.startswith('"') or return_expr.startswith("'"):
                                    return_type = '""'
                                elif return_expr.replace('.', '', 1).isdigit():
                                    return_type = return_expr  # number
                                else:
                                    return_type = 'None'
                                break
                            if lines[j].strip().startswith("def ") or lines[j].strip() == "":
                                break  # Stop scanning if new function or gap
                            j += 1
                        break

                return {
                    'action': 'insert',
                    'lines': [
                        f"{indent}# auto-fix: redefine {func_name} to avoid iterable error",
                        f"{indent}def {func_name}(*args, **kwargs):",
                        f"{indent}    return {return_type}  # mocked fallback based on return guess"
                    ]
                }

                
        
        elif 'exceptions must derive from BaseException' in err_msg:
            # Find the exception being raised
            raise_match = re.search(r'raise\s+([A-Za-z_]\w*)', line_text)
            if raise_match:
                exc_name = raise_match.group(1)
                return {'action': 'insert', 'lines': [
                    f"{indent}# auto-fix: ensure {exc_name} is a proper exception class",
                    f"{indent}if not (isinstance({exc_name}, type) and issubclass({exc_name}, BaseException)):",
                    f"{indent}    class {exc_name}(Exception):",
                    f"{indent}        def __init__(self, *args): self.args = args",
                ]}
            else:
                return {'action': 'insert', 'lines': [
                    f"{indent}# auto-fix: replace with RuntimeError",
                    f"{indent}_temp_exception = RuntimeError",
                ]}
            
        elif 'cannot unpack non-iterable' in err_msg:
            # Back up to the full multiline assignment statement
            full_expr_lines = []
            start_idx = line_no - 1
            end_idx = start_idx
            indent = ''
            
            # Collect lines upward until the start of the assignment
            while start_idx >= 0:
                line = lines[start_idx]
                full_expr_lines.insert(0, line)
                indent_match = re.match(r'^(\s*)', line)
                indent = indent_match.group(1) if indent_match else ''
                if '=' in line:
                    break
                start_idx -= 1
            
            # Collect lines downward if needed (for open parentheses)
            open_parens = sum(line.count('(') - line.count(')') for line in full_expr_lines)
            i = end_idx + 1
            while open_parens > 0 and i < len(lines):
                full_expr_lines.append(lines[i])
                open_parens += lines[i].count('(') - lines[i].count(')')
                i += 1

            # Join and extract components
            full_line = " ".join([l.strip() for l in full_expr_lines])
            unpack_match = re.match(r'([\w,\s]+)\s*=\s*(.+)', full_line)
            if unpack_match:
                lhs_vars = [v.strip() for v in unpack_match.group(1).split(',') if v.strip()]
                rhs_expr = unpack_match.group(2).strip()
                result_var = "_result"

                return {
                    'action': 'replace',
                    'lines': [
                        f"{indent}# auto-fix: safely unpack possibly None return value",
                        f"{indent}{result_var} = {rhs_expr}",
                        f"{indent}if {result_var} is None:",
                        f"{indent}    {', '.join(lhs_vars)} = " + ", ".join(["[]"] * len(lhs_vars)),
                        f"{indent}else:",
                        f"{indent}    {', '.join(lhs_vars)} = {result_var}",
                    ],
                    'replace_range': (start_idx, i)  # << NEW: custom range to replace
                }

        
        elif 'not iterable' in err_msg:
            obj_match = re.match(r"'(\w+)' object is not iterable", err_msg)
            if obj_match:
                return {'action': 'insert', 'lines': [
                    f"{indent}# auto-fix: make object iterable",
                    f"{indent}_temp_iterable = []",
                ]}
        
        elif 'unsupported operand' in err_msg:
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: ensure numeric operands",
                f"{indent}_a, _b = 0, 0  # default numeric values",
            ]}
        
            
            """
            elif 'takes no arguments' in err_msg or 'takes' in err_msg and 'arguments' in err_msg:
                var_info = extract_variable_names(line_text)
                func_calls = var_info['calls']
                if func_calls:
                    func_name = func_calls[0]
                    return {'action': 'insert', 'lines': [
                        f"{indent}# auto-fix: wrap function to accept any arguments",
                        f"{indent}_orig_{func_name} = {func_name}",
                        f"{indent}{func_name} = lambda *args, **kwargs: _orig_{func_name}() if callable(_orig_{func_name}) else None",
                    ]}
            """
            

        elif 'no len()' in err_msg:
            len_match = re.search(r'len\((\w+)\)', line_text)
            if len_match:
                var_name = len_match.group(1)
                return {'action': 'insert', 'lines': [
                    f"{indent}# auto-fix: ensure {var_name} has length",
                    f"{indent}if not hasattr({var_name}, '__len__'): {var_name} = []",
                ]}
        
        elif 'not callable' in err_msg:
            callable_match = re.search(r'([a-zA-Z_]\w*)\s*\(', line_text)
            if callable_match:
                var_name = callable_match.group(1)
                return {'action': 'insert', 'lines': [
                    f"{indent}# auto-fix: make {var_name} callable",
                    f"{indent}if not callable({var_name}): {var_name} = lambda *args, **kwargs: None",
                ]}
    
    # Handle except block errors ONLY for name/unbound errors, not all TypeErrors
    #if in_except and err_type in ['NameError', 'UnboundLocalError']:
        # Handle exceptions in 'except' blocks more intelligently
    # Handle raise inside except blocks with fallback variable setup
    if in_except and 'raise' in line_text:

        # Trace back to the corresponding try block and collect assigned vars
        try_variables = []
        try_lines = []
        for i in range(line_no - 1, -1, -1):
            line = lines[i].strip()
            if line.startswith('try:'):
                # Scan forward to gather variable assignments inside try block
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith('except'):
                    try_line = lines[j].strip()
                    try_lines.append(try_line)
                    if '=' in try_line and not try_line.startswith('#'):
                        assigns = re.findall(r'([a-zA-Z_]\w*)\s*=', try_line)
                        try_variables.extend(assigns)
                    j += 1
                break

        # Deduplicate while preserving order
        unique_vars = []
        for var in try_variables:
            if var not in unique_vars:
                unique_vars.append(var)

        # Build default assignments
        fallback_lines = [f"{indent}# auto-fix: fallback values for try-block variables"]
        for var in unique_vars:
            var_type = infer_variable_type(var, try_lines)
            default_val = create_default_value(var_type)
            fallback_lines.append(f"{indent}{var} = {default_val}")

        fallback_lines.append(f"{indent}pass  # removed raise to allow execution to continue")

        return {'action': 'replace', 'lines': fallback_lines}


        """
        if try_variables:
            # Remove duplicates while preserving order
            unique_vars = []
            for var in try_variables:
                if var not in unique_vars:
                    unique_vars.append(var)
            
            # Create assignments with inferred types
            var_assignments = []
            for var in unique_vars:
                var_type = infer_variable_type(var, try_lines)
                default_value = create_default_value(var_type)
                var_assignments.append(f"{indent}{var} = {default_value}")
            
            # Special handling for raise statements - replace with pass
            if 'raise ' in line_text:
                return {'action': 'replace', 'lines': [
                    f"{indent}# auto-fix: set try-block variables with appropriate types and skip raise",
                ] + var_assignments + [
                    f"{indent}pass  # skipped raise to continue execution"
                ]}
            else:
                # For other errors in except blocks, just set variables
                return {'action': 'insert', 'lines': [
                    f"{indent}# auto-fix: set try-block variables with appropriate types",
                ] + var_assignments}
            """
    
    var_info = extract_variable_names(line_text)
    
    # ============ ARITHMETIC ERRORS ============
    if err_type == 'ZeroDivisionError':
        return {'action': 'insert', 'lines': [
            f"{indent}# auto-fix: prevent division by zero",
            f"{indent}import sys",
            f"{indent}def safe_divide(a, b):",
            f"{indent}    return a / b if b != 0 else (float('inf') if a > 0 else float('-inf') if a < 0 else 0)",
        ]}
    
    if err_type == 'OverflowError':
        return {'action': 'insert', 'lines': [
            f"{indent}# auto-fix: handle overflow",
            f"{indent}import sys",
            f"{indent}sys.set_int_max_str_digits(0)  # Remove string conversion limit",
        ]}
    
    if err_type == 'FloatingPointError':
        return {'action': 'insert', 'lines': [
            f"{indent}# auto-fix: handle floating point error",
            f"{indent}import warnings",
            f"{indent}warnings.filterwarnings('ignore', category=RuntimeWarning)",
        ]}
    
    # ============ ATTRIBUTE ERROR ============
    if err_type == 'AttributeError':
        attr_match = re.match(r"'(\w+)' object has no attribute '(\w+)'", err_msg)
        if attr_match:
            obj_type, missing_attr = attr_match.groups()
            
            # Find the full object path (e.g., self._estimator from self._estimator.run)
            obj_path = None
            attr_pattern = rf'([a-zA-Z_][\w\.]*)\s*\.\s*{re.escape(missing_attr)}\s*\('
            path_match = re.search(attr_pattern, line_text)
            if path_match:
                obj_path = path_match.group(1)
            else:
                # Fallback: find any object.attribute pattern
                fallback_match = re.search(r'([a-zA-Z_][\w\.]*)\s*\.', line_text)
                obj_path = fallback_match.group(1) if fallback_match else 'obj'
            
            # Special handling for common cases
            if missing_attr == 'run' and obj_type == 'function':
                # Don't try to reassign 'self', handle the full path
                if obj_path == 'self':
                    # This shouldn't happen, but if it does, skip
                    return {'action': 'insert', 'lines': [
                        f"{indent}# auto-fix: cannot fix 'self' attribute error directly",
                        f"{indent}pass",
                    ]}
                else:
                    return {'action': 'insert', 'lines': [
                        f"{indent}# auto-fix: replace {obj_path} with object having run() method",
                        f"{indent}class _DummyRunner:",
                        f"{indent}    def run(self, *args, **kwargs): return None",
                        f"{indent}    def __call__(self, *args, **kwargs): return None",
                        f"{indent}if callable({obj_path}) and not hasattr({obj_path}, 'run'):",
                        f"{indent}    {obj_path} = _DummyRunner()",
                    ]}
            else:
                # For other attribute errors, use setattr approach
                if obj_path == 'self':
                    # Can't setattr on self safely, try a different approach
                    return {'action': 'insert', 'lines': [
                        f"{indent}# auto-fix: mock missing attribute {missing_attr}",
                        f"{indent}if not hasattr(self, '{missing_attr}'):",
                        f"{indent}    self.__dict__['{missing_attr}'] = lambda *args, **kwargs: None",
                    ]}
                else:
                    return {'action': 'insert', 'lines': [
                        f"{indent}# auto-fix: add missing attribute {missing_attr}",
                        f"{indent}if hasattr({obj_path.split('.')[0]}, '{obj_path.split('.')[1] if '.' in obj_path else missing_attr}'):",
                        f"{indent}    if not hasattr({obj_path}, '{missing_attr}'):",
                        f"{indent}        setattr({obj_path}, '{missing_attr}', lambda *args, **kwargs: None)",
                        f"{indent}else:",
                        f"{indent}    {obj_path.split('.')[0]}.{obj_path.split('.')[1] if '.' in obj_path else missing_attr} = type('MockObj', (), {{'{missing_attr}': lambda *a, **k: None}})()",
                    ]}
    
    # ============ IMPORT ERRORS ============
    if err_type in ['ImportError', 'ModuleNotFoundError']:
        module_match = re.match(r"No module named '(\w+)'", err_msg)
        if module_match:
            module_name = module_match.group(1)
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: create mock module {module_name}",
                f"{indent}import sys",
                f"{indent}from types import ModuleType",
                f"{indent}if '{module_name}' not in sys.modules:",
                f"{indent}    _mock_module = ModuleType('{module_name}')",
                f"{indent}    _mock_module.__dict__.update({{k: lambda *a, **kw: None for k in dir(object)}})",
                f"{indent}    sys.modules['{module_name}'] = _mock_module",
            ]}
        else:
            # Generic import error
            import_match = re.search(r'from\s+(\w+)\s+import|import\s+(\w+)', line_text)
            if import_match:
                module_name = import_match.group(1) or import_match.group(2)
                return {'action': 'insert', 'lines': [
                    f"{indent}# auto-fix: handle import error",
                    f"{indent}try:",
                    f"{indent}    pass  # original import will be attempted",
                    f"{indent}except (ImportError, ModuleNotFoundError):",
                    f"{indent}    {module_name} = type('MockModule', (), {{'__getattr__': lambda self, name: lambda *args, **kwargs: None}})()",
                ]}
    
    # ============ LOOKUP ERRORS ============
    if err_type == 'IndexError':
        if 'list index out of range' in err_msg:
            # Find list variable being indexed
            idx_vars = var_info['indexing']
            list_var = idx_vars[0] if idx_vars else 'lst'
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: ensure list has enough elements",
                f"{indent}if not isinstance({list_var}, list): {list_var} = []",
                f"{indent}while len({list_var}) < 10: {list_var}.append(None)  # ensure minimum length",
            ]}
        else:
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: handle index error with try-except",
                f"{indent}_temp_result = None",
            ]}
    
    if err_type == 'KeyError':
        key_match = re.search(r'KeyError: [\'"]?(\w+)[\'"]?', err_msg)
        if key_match:
            missing_key = key_match.group(1)
            # Find dictionary variable
            dict_vars = re.findall(r'([a-zA-Z_]\w*)\[', line_text)
            dict_var = dict_vars[0] if dict_vars else 'dct'
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: ensure key exists in dictionary",
                f"{indent}if not isinstance({dict_var}, dict): {dict_var} = {{}}",
                f"{indent}if '{missing_key}' not in {dict_var}: {dict_var}['{missing_key}'] = None",
            ]}
    
    # ============ NAME ERRORS ============
    if err_type in ['NameError', 'UnboundLocalError']:
        name_match = re.match(r"name '(\w+)' is not defined", err_msg) or \
                   re.match(r"local variable '(\w+)' referenced before assignment", err_msg)
        if name_match:
            var_name = name_match.group(1)
            
            # Special case: if it's used in a raise statement, it's likely an exception class
            if re.search(rf'raise\s+{var_name}', line_text):
                return {'action': 'insert', 'lines': [
                    f"{indent}# auto-fix: define {var_name} as exception class",
                    f"{indent}class {var_name}(Exception):",
                    f"{indent}    pass",
                ]}
            
            # Try to infer type from usage context
            elif re.search(rf'{var_name}\s*\(', line_text):
                # Used as function - but check if it might be an exception in raise context
                if 'raise' in line_text:
                    return {'action': 'insert', 'lines': [
                        f"{indent}# auto-fix: define {var_name} as exception class",
                        f"{indent}class {var_name}(Exception):",
                        f"{indent}    pass",
                    ]}
                else:
                    return {'action': 'insert', 'lines': [
                        f"{indent}# auto-fix: define {var_name} as function",
                        f"{indent}{var_name} = lambda *args, **kwargs: None",
                    ]}
            elif re.search(rf'{var_name}\[', line_text):
                # Used as indexable
                return {'action': 'insert', 'lines': [
                    f"{indent}# auto-fix: define {var_name} as list",
                    f"{indent}{var_name} = []",
                ]}
            elif re.search(rf'{var_name}\.', line_text):
                # Used as object with attributes
                return {'action': 'insert', 'lines': [
                    f"{indent}# auto-fix: define {var_name} as object",
                    f"{indent}class _TempClass: pass",
                    f"{indent}{var_name} = _TempClass()",
                ]}
            else:
                # Default to None
                return {'action': 'insert', 'lines': [
                    f"{indent}# auto-fix: define {var_name}",
                    f"{indent}{var_name} = None",
                ]}
    
    # ============ OS ERRORS ============
    if err_type == 'FileNotFoundError':
        file_match = re.search(r'[\'"]([^\'"]+)[\'"]', line_text)
        if file_match:
            filename = file_match.group(1)
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: create missing file {filename}",
                f"{indent}import os",
                f"{indent}if not os.path.exists('{filename}'):",
                f"{indent}    with open('{filename}', 'w') as f: f.write('')",
            ]}
        else:
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: handle file not found",
                f"{indent}import tempfile",
                f"{indent}_temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)",
                f"{indent}_temp_file.close()",
            ]}
    
    if err_type == 'PermissionError':
        return {'action': 'insert', 'lines': [
            f"{indent}# auto-fix: handle permission error",
            f"{indent}import tempfile, os",
            f"{indent}_temp_dir = tempfile.mkdtemp()",
            f"{indent}os.chmod(_temp_dir, 0o777)",
        ]}
    
    if err_type == 'ConnectionError':
        return {'action': 'insert', 'lines': [
            f"{indent}# auto-fix: mock network connection",
            f"{indent}class _MockResponse:",
            f"{indent}    def __init__(self): self.status_code = 200; self.text = ''; self.content = b''",
            f"{indent}    def json(self): return {{}}",
            f"{indent}_mock_response = _MockResponse()",
        ]}
    
    # ============ VALUE ERRORS ============
    if err_type == 'ValueError':
        if 'not enough values to unpack' in err_msg:
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: provide enough values for unpacking",
                f"{indent}_temp_values = [None] * 10  # provide default values",
            ]}
        elif 'too many values to unpack' in err_msg:
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: limit values for unpacking",
                f"{indent}_temp_limited = [][:2]  # limit to expected number",
            ]}
        elif 'invalid literal' in err_msg:
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: handle invalid literal conversion",
                f"{indent}_temp_value = 0  # default numeric value",
            ]}
    
    # ============ RECURSION ERROR ============
    if err_type == 'RecursionError':
        func_calls = var_info['calls']
        if func_calls:
            func_name = func_calls[0]
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: prevent recursion",
                f"{indent}{func_name} = lambda *args, **kwargs: None",
            ]}
        else:
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: recursion limit reached",
                f"{indent}import sys",
                f"{indent}sys.setrecursionlimit(10000)",
            ]}
    
    # ============ INDENTATION ERROR ============
    if err_type == 'IndentationError':
        return {'action': 'insert', 'lines': [
            f"{indent}pass  # auto-fix: indentation error"
        ]}
    
    # ============ SYSTEM EXIT ============
    if err_type == 'SystemExit':
        return {'action': 'insert', 'lines': [
            f"{indent}# auto-fix: prevent system exit",
            f"{indent}import sys",
            f"{indent}_orig_exit = sys.exit",
            f"{indent}sys.exit = lambda *args: None",
        ]}
    
    # ============ FALLBACK ============
    """
    return {'action': 'insert', 'lines': [
        f'{indent}# auto-fix: unhandled {err_type} - adding generic try-except',
        f'{indent}try:',
        f'{indent}    pass  # original code will execute in try block',
        f'{indent}except {err_type}:',
        f'{indent}    pass  # ignore this specific error type',
    ]}"""
    

def auto_fix_loop(target_path: Path):
    max_iterations = 50  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"Iteration {iteration}...")
        
        proc = subprocess.run([sys.executable, str(target_path)], capture_output=True, text=True)

        if proc.returncode == 0:
            print(f"\n✅ {target_path.name} ran successfully after {iteration} iterations!")
            break

        print(proc.stderr)
        print(target_path)
        parsed = parse_traceback(proc.stderr, target_path)
        print(parsed)
        if not parsed:
            print("Could not parse traceback; aborting.")
            print("STDERR:", proc.stderr)
            break

        err_type, err_msg, line_no = parsed
        print(f"Detected {err_type} at line {line_no}: {err_msg}")

        lines = target_path.read_text().splitlines()
        idx = (line_no - 1) if line_no and line_no > 0 else 0
        line_text = lines[idx] if idx < len(lines) else ''
        
        print(f"Error line: {line_text.strip()}")

        fix = make_fix(err_type, err_msg, line_text, lines, line_no)

        if isinstance(fix, dict):
            if fix.get('action') == 'replace':
                replace_start, replace_end = fix.get('replace_range', (idx, idx + 1))
                lines[replace_start:replace_end] = fix['lines']
                insert_offset = len(fix['lines'])
            elif fix.get('action') == 'insert':
                lines[idx:idx] = fix['lines']
                insert_offset = len(fix['lines'])
            else:
                insert_offset = 0

            if 'lines_after' in fix and isinstance(fix['lines_after'], list):
                lines[idx + insert_offset + 1:idx + insert_offset + 1] = fix['lines_after']

        target_path.write_text("\n".join(lines) + "\n")
        print(f"Applied {fix['action']} at line {line_no}")
        print("Fix applied:")
        for line in fix['lines']:
            print(f"  + {line}")
        print("Retrying...\n")
    
    if iteration >= max_iterations:
        print(f"⚠️  Maximum iterations ({max_iterations}) reached. Manual intervention may be required.")


def auto_fix(file_path):
    script_path = Path(file_path)
    if not script_path.is_file():
        print(f"Error: {script_path} not found.")
        sys.exit(1)

    auto_fix_loop(script_path)



if __name__ == "__main__":
    file_path = os.path.abspath("C:/Users/rical/OneDrive/Desktop/QSmell_Tool/generated_executables/executable__run.py")
    auto_fix(file_path)