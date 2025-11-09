def make_fix(err_type, err_msg, line_text, lines, line_no):
    """
    Return Python code lines to insert before the error line,
    given the error type, message, and the source line text.
    Also receives the full lines array and line number for context analysis.
    """
    indent_match = re.match(r'(\s*)', line_text)
    indent = indent_match.group(1) if indent_match else ''

    if debug: print(f"Error type: {err_type}")
    
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
        
            
            """MODIFICARE QUI"""            

            """elif 'no len()' in err_msg:
                print("Stiamo qua bello")
                len_match = re.search(r'len\((\w+)\)', line_text)
                if len_match:
                    var_name = len_match.group(1)
                    return {'action': 'insert', 'lines': [
                        f"{indent}# auto-fix: ensure {var_name} has length",
                        f"{indent}if not hasattr({var_name}, '__len__'): {var_name} = []",
                    ]}"""
        
        elif 'no len()' in err_msg:
            # Extract the variable name from len(variable) in the error line
            len_match = re.search(r'len\((\w+)\)', line_text)
            if len_match:
                var_name = len_match.group(1)
                
                # Also look for the assignment pattern to get the result variable
                assignment_match = re.search(r'(\w+)\s*=.*len\(' + var_name + r'\)', line_text)
                result_var = assignment_match.group(1) if assignment_match else None
                
                # Check if this looks like a numpy array context (common patterns)
                is_numpy_context = any(pattern in line_text for pattern in ['np.', 'numpy', 'log2', 'array', 'state'])
                
                if is_numpy_context:
                    return {'action': 'replace', 'lines': [
                        f"{indent}# auto-fix: replace mock with actual numpy array using original object as element",
                        f"{indent}import numpy as np",
                        f"{indent}{var_name} = []  # wrap original object in numpy array",
                        f"{indent}# Direct assignment for size calculation to avoid further errors",
                    ] + ([f"{indent}{result_var} = 0  # default size value"] if result_var else [])}
                else:
                    # Generic case - provide a list with the original object as element
                    return {'action': 'replace', 'lines': [
                        f"{indent}# auto-fix: replace problematic object with list containing itself",
                        f"{indent}{var_name} = []  # wrap original object in list",
                    ] + ([f"{indent}{result_var} = 0  # default length value"] if result_var else [])}
            else:
                # Fallback when we can't parse the len() call
                return {'action': 'replace', 'lines': [
                    f"{indent}# auto-fix: generic len() error - provide default list",
                    f"{indent}_temp_list = [0, 1]",
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
            
            # Find the full object path (e.g., self.ansatz from self.ansatz.parameters)
            obj_path = None
            attr_pattern = rf'([a-zA-Z_][\w\.]*)\s*\.\s*{re.escape(missing_attr)}'
            path_match = re.search(attr_pattern, line_text)
            if path_match:
                obj_path = path_match.group(1)
            else:
                # Fallback: find any object.attribute pattern
                fallback_match = re.search(r'([a-zA-Z_][\w\.]*)\s*\.', line_text)
                obj_path = fallback_match.group(1) if fallback_match else 'obj'
            
            # Determine where to insert the fix
            insertion_idx, indent = get_fix_insertion_point(lines, line_no - 1)
            
            # Special handling for 'parameters' attribute (generalized from ansatz-specific)
            if missing_attr == 'parameters':
                # Extract the object name from the path (e.g., 'ansatz' from 'self.ansatz')
                obj_name = obj_path.split('.')[-1] if '.' in obj_path else obj_path
                
                return {'action': 'insert_at_position', 
                    'target_line': insertion_idx,
                    'lines': [
                    f"{indent}# auto-fix: ensure {obj_path} has {missing_attr} attribute",
                    f"{indent}if hasattr({'.'.join(obj_path.split('.')[:-1]) if '.' in obj_path else 'self'}, '{obj_name}') and {obj_path} is not None:",
                    f"{indent}    if not hasattr({obj_path}, '{missing_attr}'):",
                    f"{indent}        # Create mock parameters that can be iterated",
                    f"{indent}        {obj_path}.{missing_attr} = ['param_0', 'param_1', 'param_2']",
                    f"{indent}else:",
                    f"{indent}    # Create mock {obj_name} with {missing_attr}",
                    f"{indent}    class Mock{obj_name.capitalize()}:",
                    f"{indent}        def __init__(self): self.{missing_attr} = ['param_0', 'param_1', 'param_2']",
                    f"{indent}    {obj_path} = Mock{obj_name.capitalize()}()",
                ]}
            
            # Special handling for common iterable attributes
            elif missing_attr in ['items', 'keys', 'values', 'data', 'results']:
                return {'action': 'insert_at_position',
                    'target_line': insertion_idx,
                    'lines': [
                    f"{indent}# auto-fix: ensure {obj_path} has {missing_attr} attribute",
                    f"{indent}if not hasattr({obj_path}, '{missing_attr}'):",
                    f"{indent}    # Create mock {missing_attr} method/attribute",
                    f"{indent}    if '{missing_attr}' in ['items', 'keys', 'values']:",
                    f"{indent}        setattr({obj_path}, '{missing_attr}', lambda: [])",
                    f"{indent}    else:",
                    f"{indent}        setattr({obj_path}, '{missing_attr}', [])",
                ]}
            
            # Special handling for self attributes
            elif obj_path == 'self':
                return {'action': 'insert_at_position',
                    'target_line': insertion_idx,
                    'lines': [
                    f"{indent}# auto-fix: mock missing attribute {missing_attr}",
                    f"{indent}if not hasattr(self, '{missing_attr}'):",
                    f"{indent}    self.__dict__['{missing_attr}'] = lambda *args, **kwargs: None",
                ]}
            
            # Generic attribute fix - works for any object and any attribute
            else:

                # Determine if we need a callable or simple value based on context
                is_callable = '(' in line_text[line_text.find(missing_attr):line_text.find(missing_attr) + 50] if missing_attr in line_text else False
                
                # Get smart mock value using dynamic detection
                mock_result = get_smart_mock_value(missing_attr, is_callable)
                mock_value = mock_result  # This will be the lambda or "1"

                """
                # Determine if we need a callable or simple value based on context
                is_callable = '(' in line_text[line_text.find(missing_attr):line_text.find(missing_attr) + 50] if missing_attr in line_text else False
                #mock_value = "lambda *args, **kwargs: None" if is_callable else "None"
                mock_value = "lambda *args, **kwargs: None" if is_callable else "1"
                
                # Split the object path to get base object and attribute chain
                path_parts = obj_path.split('.')
                base_obj = path_parts[0]
                
                if len(path_parts) == 1:
                    # Simple case: obj.attr
                    return {'action': 'insert_at_position',
                        'target_line': insertion_idx,
                        'lines': [
                        f"{indent}# auto-fix: add missing attribute {missing_attr}",
                        f"{indent}if not hasattr({obj_path}, '{missing_attr}'):",
                        f"{indent}    setattr({obj_path}, '{missing_attr}', {mock_value})",
                    ]}
                """

               
                
                # Split the object path to get base object and attribute chain
                path_parts = obj_path.split('.')
                base_obj = path_parts[0]
                
                # Handle dynamic type redefinition
                if mock_result.startswith('REDEFINE_AS_'):
                    # Extract the detected type info
                    detected_type, sample_value, import_stmt = find_method_in_common_types(missing_attr)
                    
                    fix_lines = [
                        f"{indent}# auto-fix: redefine {base_obj} as {detected_type} for attribute {missing_attr}",
                    ]
                    
                    # Add import if needed
                    if import_stmt:
                        fix_lines.append(f"{indent}{import_stmt}")
                    
                    fix_lines.extend([
                        f"{indent}if not hasattr({obj_path}, '{missing_attr}'):",
                        f"{indent}    {base_obj} = {sample_value}",
                    ])
                    
                    return {'action': 'insert_at_position',
                        'target_line': insertion_idx,
                        'lines': fix_lines}
                
                if len(path_parts) == 1:
                    # Simple case: obj.attr - create a mock object with the specific attribute
                    mock_attr_value = "lambda *args, **kwargs: None" if is_callable else "None"
                    
                    return {'action': 'insert_at_position',
                        'target_line': insertion_idx,
                        'lines': [
                            f"{indent}# auto-fix: redefine {base_obj} as mock object with attribute '{missing_attr}'",
                            f"{indent}from unittest.mock import Mock",
                            f"{indent}if not hasattr({obj_path}, '{missing_attr}'):",
                            f"{indent}    _mock = Mock()",
                            f"{indent}    _mock.{missing_attr} = {mock_attr_value}",
                            f"{indent}    {base_obj} = _mock",
                        ]}


                else:
                    # Complex case: obj.subobj.attr
                    parent_path = '.'.join(path_parts[:-1])
                    target_obj = path_parts[-1]
                    
                    return {'action': 'insert_at_position',
                        'target_line': insertion_idx,
                        'lines': [
                        f"{indent}# auto-fix: add missing attribute {missing_attr}",
                        f"{indent}if hasattr({parent_path}, '{target_obj}') and {obj_path} is not None:",
                        f"{indent}    if not hasattr({obj_path}, '{missing_attr}'):",
                        f"{indent}        setattr({obj_path}, '{missing_attr}', {mock_value})",
                        f"{indent}else:",
                        f"{indent}    # Create fallback object structure",
                        f"{indent}    {obj_path} = type('MockObj', (), {{'{missing_attr}': {mock_value}}})()",
                    ]}
        
        else:
            # Handle AttributeError with non-standard error message format
            # Try to extract object and attribute from the line_text directly
            attr_access_match = re.search(r'([a-zA-Z_][\w\.]*)\s*\.\s*([a-zA-Z_]\w*)', line_text)
            if attr_access_match:
                obj_path, missing_attr = attr_access_match.groups()
                
                # Determine where to insert the fix
                insertion_idx, indent = get_fix_insertion_point(lines, line_no - 1)
                
                # Generic fallback for any attribute access
                is_callable = '(' in line_text[line_text.find(missing_attr):line_text.find(missing_attr) + 50] if missing_attr in line_text else False
                mock_value = "lambda *args, **kwargs: None" if is_callable else "None"
                
                return {'action': 'insert_at_position',
                    'target_line': insertion_idx,
                    'lines': [
                    f"{indent}# auto-fix: handle non-standard AttributeError",
                    f"{indent}if not hasattr({obj_path}, '{missing_attr}'):",
                    f"{indent}    setattr({obj_path}, '{missing_attr}', {mock_value})",
                ]}
            
            # If we can't parse anything useful, create a generic attribute handler
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: generic AttributeError handler",
                f"{indent}# Create a mock object to prevent further attribute errors",
                #f"{indent}_mock_obj = type('MockObj', (), {'__getattr__': lambda self, name: lambda *args, **kwargs: None})()",
                f"{indent}_mock_obj = type('MockObj', (), {{'__getattr__': lambda self, name: lambda *args, **kwargs: None}})()"

            ]}

    """
    # ============ IMPORT ERRORS ============
    if err_type in ['ImportError', 'ModuleNotFoundError']:
        module_match = re.match(r"No module named '(\w+)'", err_msg)
        if module_match:
            module_name = module_match.group(1)
            # Test if the module can be imported
            try:
                __import__(module_name)
            except (ImportError, ModuleNotFoundError):
                print(f"ERROR: Required library '{module_name}' is not installed.")
                print(f"Please install it using: pip install {module_name}")
                print("Stopping fix process...")
                sys.exit(1)
            
            # If import is successful, continue without adding any fix
            return None  # or however you indicate "no action needed"
        else:
            # Generic import error
            import_match = re.search(r'from\s+(\w+)\s+import|import\s+(\w+)', line_text)
            if import_match:
                module_name = import_match.group(1) or import_match.group(2)
                
                # Test if the module can be imported
                try:
                    __import__(module_name)
                except (ImportError, ModuleNotFoundError):
                    print(f"ERROR: Required library '{module_name}' is not installed.")
                    print(f"Please install it using: pip install {module_name}")
                    print("Stopping fix process...")
                    exit(1)
                
                # If import is successful, continue without adding any fix
                return None  # or however you indicate "no action needed"
    """
    
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
    
    if err_type == 'SyntaxError':
        return {'action': 'delete', 'error_type': err_type, 'error_msg': err_msg}
    



    # Add this section in your make_fix function, before the fallback
    # ============ QISKIT ERRORS ============
    if err_type == 'qiskit.exceptions.QiskitError':
        if 'Invalid input data for Pauli' in err_msg:
            # This error often occurs when passing invalid data to Pauli operators
            # Find the variable being passed to init_observable or similar functions
            func_call_match = re.search(r'(\w+)\s*=\s*\w+\(([^)]+)\)', line_text)
            if func_call_match:
                result_var, input_var = func_call_match.groups()
                input_var = input_var.strip()
                return {'action': 'insert', 'lines': [
                    f"{indent}# auto-fix: ensure valid Pauli input data",
                    f"{indent}if {input_var} is None or (isinstance({input_var}, str) and {input_var} not in ['I', 'X', 'Y', 'Z']):",
                    f"{indent}    {input_var} = 'I'  # Default to identity Pauli",
                    f"{indent}elif hasattr({input_var}, '__iter__') and not isinstance({input_var}, str):",
                    f"{indent}    {input_var} = ['I'] * len({input_var}) if len({input_var}) > 0 else ['I']",
                ]}
        
        elif 'Circuit and parameter mismatch' in err_msg:
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: handle circuit parameter mismatch",
                f"{indent}# Create a simple circuit with basic gates",
                f"{indent}from qiskit import QuantumCircuit",
                f"{indent}_temp_circuit = QuantumCircuit(2)",
                f"{indent}_temp_circuit.h(0)",
                f"{indent}_temp_circuit.cx(0, 1)",
            ]}
        
        elif 'Backend' in err_msg or 'provider' in err_msg.lower():
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: use simulator backend",
                f"{indent}from qiskit import Aer",
                f"{indent}_backend = Aer.get_backend('qasm_simulator')",
            ]}
        
        # Generic Qiskit error fallback
        else:
            return {'action': 'insert', 'lines': [
                f"{indent}# auto-fix: generic Qiskit error - use mock values",
                f"{indent}try:",
                f"{indent}    pass  # original Qiskit operation",
                f"{indent}except Exception as e:",
                f"{indent}    if debug: print(f'Qiskit error bypassed: {{e}}')",
                f"{indent}    # Provide mock result based on context",
                f"{indent}    if 'observable' in locals():",
                f"{indent}        converted_observable = 'I'",
                f"{indent}    if 'circuit' in locals():",
                f"{indent}        from qiskit import QuantumCircuit",
                f"{indent}        circuit = QuantumCircuit(2)",
            ]}
    
    
    # ============ FALLBACK ============
    # Signal that we hit an unhandled error and should stop
    # return {'action': 'stop', 'error_type': err_type, 'error_msg': err_msg}
    #return {'action': 'delete', 'error_type': err_type, 'error_msg': err_msg}

    # ============ FALLBACK ============
    # Use smart deletion for unhandled errors
    idx = (line_no - 1) if line_no and line_no > 0 else 0
    if idx >= len(lines):
        return {'action': 'stop', 'error_type': err_type, 'error_msg': 'Line index out of range'}

    # Use smart deletion for multi-line statements  
    modified_lines, deleted_content, lines_deleted = smart_delete_statement(lines.copy(), idx)

    return {
        'action': 'smart_delete',
        'error_type': err_type,
        'error_msg': err_msg,
        'modified_lines': modified_lines,
        'deleted_content': deleted_content,
        'lines_deleted': lines_deleted
    }
 