#!/usr/bin/env python3
import subprocess
import sys
import re
import os
from pathlib import Path



debug=True

def get_fix_insertion_point(lines, error_line_idx):
    """
    Find the best insertion point for a fix before a problematic line.
    Returns (insertion_index, indentation_string).
    
    This version prioritizes finding assignment statements which are 
    typically where we want to insert attribute fixes.
    """
    
    # Look backwards for assignment patterns - start close and expand search
    for i in range(error_line_idx, max(-1, error_line_idx - 25), -1):
        line = lines[i].strip()
        if not line:
            continue
            
        # Look for assignment patterns first - this is usually what we want
        if '=' in line and not line.startswith('==') and not line.startswith('!='):
            # Simple check to avoid assignments inside strings
            equals_pos = line.find('=')
            before_equals = line[:equals_pos]
            
            # Count quotes before the equals to see if we're in a string
            double_quotes = before_equals.count('"')
            single_quotes = before_equals.count("'")
            
            # If even number of quotes, the = is not inside a string
            if double_quotes % 2 == 0 and single_quotes % 2 == 0:
                # This looks like a real assignment statement
                indent_match = re.match(r'^(\s*)', lines[i])
                indent = indent_match.group(1) if indent_match else ''
                return i, indent
        
        # Also look for other statement starters, but with less priority
        if any(line.startswith(keyword) for keyword in ['def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ']):
            indent_match = re.match(r'^(\s*)', lines[i])
            indent = indent_match.group(1) if indent_match else ''
            return i, indent
    
    # If no assignment found, try the original functions as fallback
    try:
        # Try declaration start detection
        decl_start = find_declaration_start(lines, error_line_idx)
        if decl_start is not None:
            decl_line = lines[decl_start]
            indent_match = re.match(r'^(\s*)', decl_line)
            indent = indent_match.group(1) if indent_match else ''
            return decl_start, indent
        
        # Try complete statement range detection
        start_idx, _ = find_complete_statement_range(lines, error_line_idx)
        if start_idx < error_line_idx:
            start_line = lines[start_idx]
            indent_match = re.match(r'^(\s*)', start_line)
            indent = indent_match.group(1) if indent_match else ''
            return start_idx, indent
            
    except Exception as e:
        # If the other functions fail, continue to fallback
        pass
    
    # Final fallback: insert before the error line itself
    error_line = lines[error_line_idx]
    indent_match = re.match(r'^(\s*)', error_line)
    indent = indent_match.group(1) if indent_match else ''
    return error_line_idx, indent

def find_complete_statement_range(lines, error_line_idx):
    """
    Find the complete range of a multi-line statement that includes the error line.
    Returns (start_idx, end_idx) where both are inclusive.
    """
    start_idx = error_line_idx
    end_idx = error_line_idx
    
    # Get the base indentation of the error line
    error_line = lines[error_line_idx].rstrip()
    base_indent_match = re.match(r'^(\s*)', error_line)
    base_indent = base_indent_match.group(1) if base_indent_match else ''
    
    # Look backwards to find the start of the statement
    # We need to find where the assignment or function call begins
    for i in range(error_line_idx, -1, -1):
        line = lines[i].rstrip()
        if not line.strip():  # Skip empty lines
            continue
            
        line_indent_match = re.match(r'^(\s*)', line)
        line_indent = line_indent_match.group(1) if line_indent_match else ''
        
        # If we find a line with less indentation, we've gone too far
        if len(line_indent) < len(base_indent):
            break
            
        # If we find a line that starts a new statement (contains = or is a function def, etc.)
        # and has the same indentation level, this might be our start
        if len(line_indent) == len(base_indent):
            stripped = line.strip()
            # Check if this line starts a statement
            if ('=' in stripped and not stripped.startswith('==') or 
                stripped.endswith('(') or 
                any(stripped.startswith(keyword) for keyword in ['def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'with '])):
                start_idx = i
                break
        
        start_idx = i
    
    # Look forwards to find the end of the statement
    # We need to balance parentheses, brackets, and braces
    paren_count = 0
    bracket_count = 0
    brace_count = 0
    
    # Count opening/closing characters from start to current position
    for i in range(start_idx, error_line_idx + 1):
        line = lines[i]
        paren_count += line.count('(') - line.count(')')
        bracket_count += line.count('[') - line.count(']')
        brace_count += line.count('{') - line.count('}')
    
    # Continue forward until all brackets are balanced
    for i in range(error_line_idx + 1, len(lines)):
        line = lines[i].rstrip()
        if not line.strip():  # Skip empty lines
            end_idx = i
            continue
            
        line_indent_match = re.match(r'^(\s*)', line)
        line_indent = line_indent_match.group(1) if line_indent_match else ''
        
        # Update bracket counts
        paren_count += line.count('(') - line.count(')')
        bracket_count += line.count('[') - line.count(']')
        brace_count += line.count('{') - line.count('}')
        
        end_idx = i
        
        # If all brackets are balanced and we're at the same or less indentation
        if (paren_count <= 0 and bracket_count <= 0 and brace_count <= 0):
            # Check if this line ends the statement
            stripped = line.strip()
            if (stripped.endswith(')') or stripped.endswith(']') or stripped.endswith('}') or
                stripped.endswith(';') or not stripped.endswith((',', '\\'))):
                break
        
        # If we encounter a line with less indentation and balanced brackets, stop
        if (len(line_indent) < len(base_indent) and 
            paren_count <= 0 and bracket_count <= 0 and brace_count <= 0):
            end_idx = i - 1
            break
    
    return start_idx, end_idx


def extract_class_from_assignment(deleted_lines):
    """
    Extract the class name from a deleted assignment.
    Returns the class name or None if not found.
    """
    # Join all deleted lines to get the complete statement
    full_statement = ' '.join(line.strip() for line in deleted_lines)
    
    # Remove extra whitespace and newlines
    full_statement = re.sub(r'\s+', ' ', full_statement).strip()
    
    # Pattern 1: Direct class instantiation - var = ClassName(...)
    # This handles: job = AlgorithmJob(...), result = SomeClass(...), etc.
    class_match = re.search(r'=\s*([A-Z][a-zA-Z0-9_]*(?:\.[A-Z][a-zA-Z0-9_]*)*)\s*\(', full_statement)
    if class_match:
        return class_match.group(1)
    
    # Pattern 2: Module.ClassName(...) - handles imported classes
    # This handles: job = qiskit_algorithms.AlgorithmJob(...), etc.  
    module_class_match = re.search(r'=\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*\.[A-Z][a-zA-Z0-9_]*)\s*\(', full_statement)
    if module_class_match:
        full_class_path = module_class_match.group(1)
        # Return just the class name (last part after the last dot)
        return full_class_path.split('.')[-1]
    
    # Pattern 3: Function calls that might return class instances
    # This is a fallback for functions that create objects
    func_match = re.search(r'=\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\(', full_statement)
    if func_match:
        func_name = func_match.group(1)
        
        # Try to infer the class from function name patterns
        if 'create' in func_name.lower() or 'make' in func_name.lower():
            # Functions like create_job, make_circuit, etc.
            if 'job' in func_name.lower():
                return 'Job'
            elif 'circuit' in func_name.lower():
                return 'Circuit'  
            elif 'backend' in func_name.lower():
                return 'Backend'
            elif 'result' in func_name.lower():
                return 'Result'
        
        # Check if function name suggests return type
        elif func_name.endswith('Job') or 'Job' in func_name:
            return 'Job'
        elif func_name.endswith('Backend') or 'Backend' in func_name:
            return 'Backend'
        elif func_name.endswith('Circuit') or 'Circuit' in func_name:
            return 'Circuit'
        elif func_name.endswith('Result') or 'Result' in func_name:
            return 'Result'
        else:
            # Generic function call - try to extract a meaningful name
            # Take the last part of the function name and capitalize it
            base_name = func_name.split('.')[-1]
            return base_name.capitalize() if base_name else None
    
    # Pattern 4: Look for any callable that might be a class
    # This catches cases where the class name doesn't start with uppercase
    any_callable_match = re.search(r'=\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\(', full_statement)
    if any_callable_match:
        callable_name = any_callable_match.group(1)
        # Return the last part and make it a proper class name
        base_name = callable_name.split('.')[-1]
        return base_name.capitalize()
    
    return None


def create_mock_class_definition(class_name, indent):
    """
    Create a mock class definition with common methods.
    The class_name should be the actual class name extracted from the code.
    """
    mock_class_name = f"Mock{class_name}"
    
    return [
        f"{indent}# auto-fix: creating mock class for {class_name}",
        f"{indent}class {mock_class_name}:",
        f"{indent}    def __init__(self, *args, **kwargs):",
        f"{indent}        # Accept any arguments to avoid parameter errors",
        f"{indent}        pass",
        f"{indent}    ",
        f"{indent}    def __getattr__(self, name):",
        f"{indent}        # Return a callable for any method that doesn't exist",
        f"{indent}        return lambda *args, **kwargs: self",
        f"{indent}    ",
        f"{indent}    def __call__(self, *args, **kwargs):",
        f"{indent}        # Make the object callable",
        f"{indent}        return self",
        f"{indent}    ",
        f"{indent}    def __str__(self):",
        f"{indent}        return f'{mock_class_name}()'",
        f"{indent}    ",
        f"{indent}    def __repr__(self):",
        f"{indent}        return self.__str__()",
        f"{indent}    ",
        f"{indent}    # Common methods that might be called on any object",
        f"{indent}    def submit(self): return self",
        f"{indent}    def result(self): return self", 
        f"{indent}    def run(self, *args, **kwargs): return self",
        f"{indent}    def execute(self, *args, **kwargs): return self",
        f"{indent}    def get_counts(self): return {{'00': 1000, '01': 200, '10': 150, '11': 24}}",
        f"{indent}    def get_data(self): return {{}}",
        f"{indent}    def job_id(self): return 'mock_job_id'",
        f"{indent}    def status(self): return 'DONE'",
        f"{indent}    def wait_for_completion(self): return self",
        f"{indent}    def cancel(self): return True",
        f"{indent}    def backend(self): return self",
        f"{indent}    def draw(self, *args, **kwargs): return 'Mock Circuit Drawing'",
        f"{indent}    def transpile(self, *args, **kwargs): return self",
        
    ]


def find_declaration_start(lines, error_line_idx):
    """
    Find the start of a declaration/assignment that contains the error line.
    Returns the line index where the declaration starts, or None if not in a declaration.
    """
    # Look backwards from the error line to find the start of a declaration
    for i in range(error_line_idx, -1, -1):
        line = lines[i].strip()
        if not line:  # Skip empty lines
            continue
            
        # Check if this line contains an assignment operator
        if '=' in line and not line.startswith('==') and not line.startswith('!='):
            # Make sure it's not inside a string or comment
            # Simple check: if '=' appears before any quotes
            equals_pos = line.find('=')
            first_quote = min([pos for pos in [line.find('"'), line.find("'")] if pos != -1] or [len(line)])
            
            if equals_pos < first_quote:
                # This is likely a declaration line
                # Check if the error line is part of this declaration by looking at indentation
                error_line = lines[error_line_idx]
                decl_line = lines[i]
                
                # Get indentation levels
                decl_indent = len(decl_line) - len(decl_line.lstrip())
                error_indent = len(error_line) - len(error_line.lstrip())
                
                # If the error line has more indentation, it's likely part of the declaration
                if error_indent > decl_indent:
                    return i
                # If same indentation but error line doesn't start with '=', 
                # it might be a continuation
                elif error_indent == decl_indent and not error_line.strip().startswith('='):
                    # Check if there are unbalanced parentheses/brackets from declaration to error
                    paren_count = 0
                    bracket_count = 0
                    brace_count = 0
                    
                    for j in range(i, error_line_idx + 1):
                        check_line = lines[j]
                        paren_count += check_line.count('(') - check_line.count(')')
                        bracket_count += check_line.count('[') - check_line.count(']')
                        brace_count += check_line.count('{') - check_line.count('}')
                    
                    # If brackets are unbalanced, we're inside the declaration
                    if paren_count > 0 or bracket_count > 0 or brace_count > 0:
                        return i
                
                # If we found an assignment at same or higher level, we're not in a declaration
                if error_indent <= decl_indent:
                    break
        
        # If we encounter a line with less indentation that's not empty, stop searching
        if line:
            error_line = lines[error_line_idx]
            decl_indent = len(lines[i]) - len(lines[i].lstrip())
            error_indent = len(error_line) - len(error_line.lstrip())
            
            if decl_indent < error_indent:
                continue  # Keep looking backwards
            else:
                break  # We've gone too far
    
    return None  # Not inside a declaration


def is_inside_declaration(lines, error_line_idx):
    """
    Check if the error line is inside a multi-line declaration/assignment.
    Returns (True, declaration_start_idx) if inside a declaration, (False, None) otherwise.
    """
    decl_start = find_declaration_start(lines, error_line_idx)
    return (True, decl_start) if decl_start is not None else (False, None)
    """
    Intelligently delete a complete statement that spans multiple lines.
    Returns the modified lines and the number of lines deleted.
    """
    start_idx, end_idx = find_complete_statement_range(lines, error_line_idx)
    
    # Get the deleted content for logging
    deleted_lines = lines[start_idx:end_idx + 1]
    
    # Check if we're deleting a variable assignment - we might need to preserve the variable
    first_line = lines[start_idx].strip()
    assignment_match = re.match(r'^(\w+)\s*=', first_line)
    
    replacement_lines = []
    if assignment_match:
        var_name = assignment_match.group(1)
        # Get indentation from the first line
        indent_match = re.match(r'^(\s*)', lines[start_idx])
        indent = indent_match.group(1) if indent_match else ''
        
        # Try to extract the original class name
        original_class = extract_class_from_assignment(deleted_lines)
        
        if original_class:
            # Create a mock class based on the original
            mock_class_name = f"Mock{original_class}"
            
            replacement_lines = [
                f"{indent}# auto-fix: deleted problematic assignment, creating mock instance",
            ] + create_mock_class_definition(original_class, indent) + [
                f"{indent}{var_name} = {mock_class_name}()  # mock instance of {original_class}"
            ]
        else:
            # Fallback to a generic mock object
            replacement_lines = [
                f"{indent}# auto-fix: deleted problematic assignment, providing generic mock",
                f"{indent}class _GenericMock:",
                f"{indent}    def __getattr__(self, name): return lambda *args, **kwargs: self",
                f"{indent}    def __call__(self, *args, **kwargs): return self",
                f"{indent}    def __str__(self): return 'GenericMock()'",
                f"{indent}{var_name} = _GenericMock()  # generic mock instance"
            ]
    
    # Delete the problematic lines and insert replacements
    del lines[start_idx:end_idx + 1]
    if replacement_lines:
        lines[start_idx:start_idx] = replacement_lines
    
    return lines, deleted_lines, len(deleted_lines) - len(replacement_lines)


# Updated fallback in make_fix function
def make_fix_fallback(err_type, err_msg, line_text, lines, line_no):
    """
    Fallback that intelligently deletes complete multi-line statements
    """
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



def find_complete_statement_range_improved(lines, error_line_idx):
    """
    Improved version that better handles multi-line statements with proper bracket balancing.
    Returns (start_idx, end_idx) where both are inclusive.
    """
    if debug: print(f"DEBUG: find_complete_statement_range_improved called with error_line_idx={error_line_idx}")
    
    # First, find the start of the statement by looking for assignments or other statement starters
    start_idx = error_line_idx
    
    # Look backwards to find the start
    for i in range(error_line_idx, -1, -1):
        line = lines[i].strip()
        if not line:  # Skip empty lines
            continue
            
        if debug: print(f"DEBUG: Checking line {i}: '{line}'")
        
        # Check for assignment - this is usually the start of our statement
        if '=' in line and not line.startswith('==') and not line.startswith('!='):
            # Make sure it's not in a string
            equals_pos = line.find('=')
            before_equals = line[:equals_pos]
            if before_equals.count('"') % 2 == 0 and before_equals.count("'") % 2 == 0:
                if debug: print(f"DEBUG: Found assignment start at line {i}")
                start_idx = i
                break
        
        # Check for other statement starters
        if any(line.startswith(keyword) for keyword in ['def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ', 'return ']):
            if debug: print(f"DEBUG: Found statement start at line {i}")
            start_idx = i
            break
            
        # If we find a line with significantly less indentation, stop
        if line:
            current_indent = len(lines[i]) - len(lines[i].lstrip())
            error_indent = len(lines[error_line_idx]) - len(lines[error_line_idx].lstrip())
            
            if current_indent < error_indent - 4:  # Allow some flexibility
                if debug: print(f"DEBUG: Significant indentation change at line {i}, stopping")
                break
                
        start_idx = i
    
    if debug: print(f"DEBUG: Statement starts at line {start_idx}")
    
    # Now find the end by balancing brackets from the start
    end_idx = error_line_idx
    
    # Count all brackets from the start line
    paren_count = 0
    bracket_count = 0
    brace_count = 0
    
    # Start counting from the beginning of the statement
    for i in range(start_idx, len(lines)):
        line = lines[i]
        
        # Update bracket counts
        line_paren = line.count('(') - line.count(')')
        line_bracket = line.count('[') - line.count(']')
        line_brace = line.count('{') - line.count('}')
        
        paren_count += line_paren
        bracket_count += line_bracket
        brace_count += line_brace
        
        if debug: print(f"DEBUG: Line {i}: '{line.strip()}' | Parens: {paren_count}, Brackets: {bracket_count}, Braces: {brace_count}")
        
        end_idx = i
        
        # If we're past the error line and all brackets are balanced
        if i >= error_line_idx and paren_count <= 0 and bracket_count <= 0 and brace_count <= 0:
            if debug: print(f"DEBUG: All brackets balanced at line {i}")
            break
            
        # Safety check: if we've gone too far past the error line and brackets are still unbalanced
        if i > error_line_idx + 20:
            if debug: print(f"DEBUG: Safety break at line {i}")
            break
    
    if debug: print(f"DEBUG: Statement ends at line {end_idx}")
    return start_idx, end_idx

def smart_delete_statement(lines, error_line_idx):
    """
    Improved version that uses better range detection.
    Returns the modified lines, deleted content, and number of lines deleted.
    """
    if debug: print(f"DEBUG: smart_delete_statement_improved called with error_line_idx={error_line_idx}")
    
    # Use the improved range finder
    start_idx, end_idx = find_complete_statement_range_improved(lines, error_line_idx)
    
    if debug: print(f"DEBUG: Will delete lines {start_idx} to {end_idx} (inclusive)")
    
    # Get the deleted content for logging
    deleted_lines = lines[start_idx:end_idx + 1]
    
    if debug: print("DEBUG: Lines to be deleted:")
    for i, line in enumerate(deleted_lines):
        if debug: print(f"  {start_idx + i}: '{line.rstrip()}'")
    
    # Check if we're deleting a variable assignment - we might need to preserve the variable
    first_line = lines[start_idx].strip()
    assignment_match = re.match(r'^([a-zA-Z_][\w\.]*)\s*=', first_line)
    
    replacement_lines = []
    if assignment_match:
        var_name = assignment_match.group(1)
        if debug: print(f"DEBUG: Found variable assignment: {var_name}")
        
        # Get indentation from the first line
        indent_match = re.match(r'^(\s*)', lines[start_idx])
        indent = indent_match.group(1) if indent_match else ''
        
        # Try to extract the original class name
        original_class = extract_class_from_assignment(deleted_lines)
        
        if original_class:
            if debug: print(f"DEBUG: Extracted class name: {original_class}")
            # Create a mock class based on the original
            mock_class_name = f"Mock{original_class}"
            
            replacement_lines = [
                f"{indent}# auto-fix: deleted problematic assignment, creating mock instance",
            ] + create_mock_class_definition(original_class, indent) + [
                f"{indent}{var_name} = {mock_class_name}()  # mock instance of {original_class}"
            ]
        else:
            if debug: print("DEBUG: No class name extracted, using generic mock")
            # Fallback to a generic mock object
            replacement_lines = [
                f"{indent}# auto-fix: deleted problematic assignment, providing generic mock",
                f"{indent}class _GenericMock:",
                f"{indent}    def __getattr__(self, name): return lambda *args, **kwargs: self",
                f"{indent}    def __call__(self, *args, **kwargs): return self",
                f"{indent}    def __str__(self): return 'GenericMock()'",
                f"{indent}{var_name} = _GenericMock()  # generic mock instance"
            ]
    #else:
        if debug: print("DEBUG: No variable assignment found, just deleting")
    
    # Delete the problematic lines and insert replacements
    if debug: print(f"DEBUG: Deleting lines {start_idx}:{end_idx + 1}")
    del lines[start_idx:end_idx + 1]
    
    if replacement_lines:
        if debug: print(f"DEBUG: Inserting {len(replacement_lines)} replacement lines at position {start_idx}")
        lines[start_idx:start_idx] = replacement_lines
    
    lines_deleted_count = len(deleted_lines) - len(replacement_lines)
    if debug: print(f"DEBUG: Net lines deleted: {lines_deleted_count}")
    
    return lines, deleted_lines, lines_deleted_count






























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
    # Updated regex to handle module-prefixed exceptions
    exc_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_.]*(?:Error|Exception)): (.+)', exc_line)
    if not exc_match:
        # Try to match just the error type without message
        exc_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_.]*(?:Error|Exception))$', exc_line)
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
                mock_value = "lambda *args, **kwargs: None" if is_callable else "None"
                
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
                exit(1)
            
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
        print("Ci troviamo qui")
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
    




import subprocess
import sys
import re
from pathlib import Path

def parse_syntax_error(stderr_output, target_path):
    """
    Parse syntax errors from Python stderr output.
    Returns (error_type, error_message, line_number) or None if not a syntax error.
    """
    # Look for syntax error patterns
    syntax_patterns = [
        # Standard syntax error format
        r'File "([^"]+)", line (\d+)\s*\n\s*(.+)\n\s*\^\s*\nSyntaxError: (.+)',
        # Alternative syntax error format
        r'SyntaxError: (.+) \(([^,]+), line (\d+)\)',
        # Another common format
        r'File "([^"]+)", line (\d+)\s*.*\n.*SyntaxError: (.+)',
    ]
    
    for pattern in syntax_patterns:
        match = re.search(pattern, stderr_output, re.MULTILINE | re.DOTALL)
        if match:
            if len(match.groups()) >= 3:
                # Extract line number (could be in different positions depending on pattern)
                if pattern == syntax_patterns[1]:  # Special case for format 2
                    error_msg = match.group(1)
                    line_no = int(match.group(3))
                else:
                    line_no = int(match.group(2))
                    error_msg = match.group(-1)  # Last group is usually the error message
                
                return 'SyntaxError', error_msg.strip(), line_no
    
    # Try a simpler approach - look for any line with "SyntaxError" and extract line number
    lines = stderr_output.split('\n')
    for i, line in enumerate(lines):
        if 'SyntaxError:' in line:
            # Look for line number in previous lines
            for j in range(max(0, i-3), i):
                line_match = re.search(r'line (\d+)', lines[j])
                if line_match:
                    line_no = int(line_match.group(1))
                    error_msg = line.split('SyntaxError:')[1].strip()
                    return 'SyntaxError', error_msg, line_no
    
    return None

def handle_syntax_error(lines, line_no, error_msg):
    """
    Handle syntax errors by removing the problematic line.
    Returns modified lines.
    """
    idx = (line_no - 1) if line_no and line_no > 0 else 0
    
    if idx >= len(lines):
        if debug: print(f"⚠️  Syntax error line {line_no} is out of range")
        return lines
    
    problematic_line = lines[idx]
    if debug: print(f"⚠️  Removing syntax error line {line_no}: {problematic_line.strip()}")
    
    # Simple approach: just delete the problematic line
    del lines[idx]
    
    return lines

def check_for_syntax_errors(target_path):
    """
    Check if a Python file has syntax errors without executing it.
    Returns (has_syntax_error, error_info) where error_info is (error_type, error_msg, line_no) or None.
    """
    try:
        # Try to compile the file - this will catch syntax errors without execution
        with open(target_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        compile(source_code, str(target_path), 'exec')
        return False, None
        
    except SyntaxError as e:
        # Extract syntax error information
        return True, ('SyntaxError', str(e), e.lineno)
    except Exception as e:
        # Other compilation errors
        return True, ('CompileError', str(e), None)






















































import os
import sys
import subprocess
import re
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional





import os
import sys
import subprocess
import re
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class LineMapper:
    """Tracks line number changes as code is modified during auto-fixing"""
    
    def __init__(self):
        self.mappings = {}  # Dict with structure: {(original_line, start_col, end_col): [fixed_lines]}
        self.line_offset = 0  # Current offset for line numbers
    
    def add_mapping(self, original_line: int, original_start_col: int, original_end_col: int,
                   fixed_lines: List[int]):
        """Add a mapping from original position to fixed lines"""
        key = (original_line, original_start_col, original_end_col)
        self.mappings[key] = fixed_lines
    
    def add_deletion_mapping(self, original_line: int, original_start_col: int, original_end_col: int):
        """Add a mapping for a deleted line"""
        key = (original_line, original_start_col, original_end_col)
        self.mappings[key] = []  # Empty list indicates deletion
    
    def update_line_offset(self, offset_change: int):
        """Update the line offset for subsequent operations"""
        self.line_offset += offset_change
    
    def get_adjusted_line(self, original_line: int) -> int:
        """Get the current line number accounting for all previous changes"""
        return original_line + self.line_offset
    
    def update_all_mappings_after_line(self, after_line: int, offset: int):
        """Update all existing mappings that come after a certain line due to insertions/deletions"""
        updated_mappings = {}
        for (orig_line, start_col, end_col), fixed_lines in self.mappings.items():
            # The original line number never changes, but the fixed line numbers do
            if fixed_lines:  # Only update if not deleted
                # Update fixed lines that come after the change point
                updated_fixed_lines = []
                for fixed_line in fixed_lines:
                    if fixed_line > after_line:
                        updated_fixed_lines.append(fixed_line + offset)
                    else:
                        updated_fixed_lines.append(fixed_line)
                updated_mappings[(orig_line, start_col, end_col)] = updated_fixed_lines
            else:
                # Keep deleted mappings as empty lists
                updated_mappings[(orig_line, start_col, end_col)] = fixed_lines
        self.mappings = updated_mappings


"""
def extract_main_function_lines(file_path: Path) -> Dict[int, Tuple[int, int]]:
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        main_lines = {}
        
        in_main_block = False
        main_indent_level = None
        
        for line_no, line in enumerate(lines, 1):
            stripped_line = line.strip()
            
            # Check for main block start
            if stripped_line.startswith('if __name__') and '__main__' in stripped_line:
                in_main_block = True
                # Find the base indentation level for the main block
                main_indent_level = len(line) - len(line.lstrip())
                # Include the if __name__ line itself
                start_col = len(line) - len(line.lstrip())
                end_col = len(line.rstrip())
                main_lines[line_no] = (start_col, end_col)
                continue
            
            if in_main_block:
                # Skip empty lines and comments (but still include them in mapping)
                if not stripped_line:
                    main_lines[line_no] = (0, 0)  # Empty line
                    continue
                    
                if stripped_line.startswith('#'):
                    start_col = line.find('#')
                    end_col = len(line.rstrip())
                    main_lines[line_no] = (start_col, end_col)
                    continue
                
                # Calculate current line's indentation
                current_indent = len(line) - len(line.lstrip())
                
                # If we're at or less than the main block's indentation level, we've left the main block
                if current_indent <= main_indent_level and stripped_line:
                    # Check if this is another top-level construct (class, def, etc.)
                    if any(stripped_line.startswith(keyword) for keyword in ['def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ']):
                        in_main_block = False
                        break
                
                # We're still in the main block
                if stripped_line:
                    start_col = current_indent
                    end_col = len(line.rstrip())
                    main_lines[line_no] = (start_col, end_col)
                else:
                    main_lines[line_no] = (0, 0)  # Empty line within main block
        
        return main_lines
    except Exception as e:
        print(f"Error extracting main function lines: {e}")
        return {}
"""

def extract_main_function_lines(file_path: Path) -> Dict[int, Tuple[int, int]]:
    """Extract line numbers and column positions of all lines in the file"""
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        all_lines = {}
        
        for line_no, line in enumerate(lines, 1):
            stripped_line = line.strip()
            
            # Handle empty lines
            if not stripped_line:
                all_lines[line_no] = (0, 0)  # Empty line
                continue
            
            # Handle comment lines
            if stripped_line.startswith('#'):
                start_col = line.find('#')
                end_col = len(line.rstrip())
                all_lines[line_no] = (start_col, end_col)
                continue
            
            # Handle regular lines with content
            start_col = len(line) - len(line.lstrip())  # Indentation level
            end_col = len(line.rstrip())  # End of content (excluding trailing whitespace)
            all_lines[line_no] = (start_col, end_col)
        
        return all_lines
    except Exception as e:
        print(f"Error extracting file lines: {e}")
        return {}




def check_for_syntax_errors(file_path: Path) -> Tuple[bool, Optional[Tuple[str, str, int]]]:
    """Check if file has syntax errors without executing it"""
    try:
        content = file_path.read_text(encoding="utf-8")
        compile(content, str(file_path), 'exec')
        return False, None
    except SyntaxError as e:
        return True, ("SyntaxError", str(e), e.lineno)

def parse_syntax_error(stderr: str, file_path: Path) -> Optional[Tuple[str, str, int]]:
    """Parse syntax error from stderr output"""
    lines = stderr.split('\n')
    for line in lines:
        if 'SyntaxError' in line and 'line' in line:
            # Try to extract line number
            match = re.search(r'line (\d+)', line)
            if match:
                line_no = int(match.group(1))
                return ("SyntaxError", line, line_no)
    return None

def parse_traceback(stderr: str, file_path: Path) -> Optional[Tuple[str, str, int]]:
    """Parse error traceback to extract error type, message, and line number"""
    lines = stderr.split('\n')
    error_line = None
    line_no = None
    
    # Find the line with file reference
    for line in lines:
        if str(file_path) in line and 'line' in line:
            match = re.search(r'line (\d+)', line)
            if match:
                line_no = int(match.group(1))
                break
    
    # Find the error type and message
    for line in reversed(lines):
        if line.strip() and ':' in line and any(err in line for err in ['Error', 'Exception']):
            error_line = line.strip()
            break
    
    if error_line and line_no:
        # Split error type and message
        if ':' in error_line:
            error_type = error_line.split(':')[0].strip()
            error_msg = ':'.join(error_line.split(':')[1:]).strip()
            return (error_type, error_msg, line_no)
    
    return None
"""
def make_fix(err_type: str, err_msg: str, line_text: str, lines: List[str], line_no: int) -> Optional[Dict]:
    # This is a simplified version - you'll need to implement your actual fix logic
    # Based on your code, it should return a dict with 'action' and other relevant fields
    
    if 'NameError' in err_type:
        # Example: Try to add missing import or variable definition
        return {
            'action': 'insert',
            'lines': ['# Auto-generated fix for NameError', f'# Original error: {err_msg}']
        }
    elif 'IndentationError' in err_type:
        # Fix indentation
        return {
            'action': 'replace',
            'lines': [line_text.lstrip()],  # Remove extra indentation
            'replace_range': (line_no - 1, line_no)
        }
    elif 'SyntaxError' in err_type:
        # Delete problematic line
        return {
            'action': 'delete'
        }
    else:
        # Default: comment out the problematic line
        return {
            'action': 'replace',
            'lines': [f'# {line_text.strip()} # Auto-commented due to {err_type}'],
            'replace_range': (line_no - 1, line_no)
        }
"""

def handle_syntax_error(lines: List[str], line_no: int, err_msg: str) -> List[str]:
    """Handle syntax error by removing the problematic line"""
    if line_no and 1 <= line_no <= len(lines):
        idx = line_no - 1
        del lines[idx]
    return lines

def handle_syntax_error_with_mapping(lines: List[str], line_no: int, err_msg: str, 
                                   mapper: LineMapper, main_lines: Dict[int, Tuple[int, int]]) -> List[str]:
    """Handle syntax error by removing the line and updating mapping"""
    if line_no and 1 <= line_no <= len(lines):
        idx = line_no - 1
        
        # Update mapping for deleted line if it's in main (set to empty list)
        if line_no in main_lines:
            start_col, end_col = main_lines[line_no]
            key = (line_no, start_col, end_col)
            if key in mapper.mappings:
                mapper.mappings[key] = []  # Mark as deleted
        
        # Remove the line
        del lines[idx]
        
        # Update all mappings after this line (shift up by 1)
        mapper.update_all_mappings_after_line(line_no, -1)
        mapper.update_line_offset(-1)
    
    return lines

def apply_fix_with_mapping(lines: List[str], fix: Dict, line_no: int, 
                         mapper: LineMapper, main_lines: Dict[int, Tuple[int, int]]) -> List[str]:
    """Apply fix and update line mappings"""
    idx = (line_no - 1) if line_no and line_no > 0 else 0
    
    if isinstance(fix, dict):
        action = fix.get('action')
        
        if action == 'insert_at_position':
            target_idx = fix['target_line']
            insert_lines = fix['lines']
            
            # Insert the new lines
            lines[target_idx:target_idx] = insert_lines
            
            # Update all mappings after the insertion point
            lines_added = len(insert_lines)
            mapper.update_all_mappings_after_line(target_idx, lines_added)
            mapper.update_line_offset(lines_added)
            
        elif action == 'insert_before_statement':
            target_idx = fix['target_line']
            insert_lines = fix['lines']
            
            lines[target_idx:target_idx] = insert_lines
            lines_added = len(insert_lines)
            
            # Update all mappings after the insertion point
            mapper.update_all_mappings_after_line(target_idx, lines_added)
            mapper.update_line_offset(lines_added)
            
        elif action == 'smart_delete':
            if 'modified_lines' in fix:
                # Use pre-computed modified lines
                lines[:] = fix['modified_lines']
                lines_deleted = fix.get('lines_deleted', 0)
                
                # Update mappings for deleted lines
                for i in range(lines_deleted):
                    original_line_num = line_no + i
                    if original_line_num in main_lines:
                        start_col, end_col = main_lines[original_line_num]
                        key = (original_line_num, start_col, end_col)
                        if key in mapper.mappings:
                            mapper.mappings[key] = []  # Mark as deleted
                
                # Update all mappings after the deletion
                mapper.update_all_mappings_after_line(line_no + lines_deleted - 1, -lines_deleted)
                mapper.update_line_offset(-lines_deleted)
            else:
                # Fallback to single line deletion
                if idx < len(lines):
                    if line_no in main_lines:
                        start_col, end_col = main_lines[line_no]
                        key = (line_no, start_col, end_col)
                        if key in mapper.mappings:
                            mapper.mappings[key] = []  # Mark as deleted
                    
                    del lines[idx]
                    mapper.update_all_mappings_after_line(line_no, -1)
                    mapper.update_line_offset(-1)
                
        elif action == 'delete':
            # Single line deletion
            if idx < len(lines):
                # Update mapping for deleted line if in main (set to empty list)
                if line_no in main_lines:
                    start_col, end_col = main_lines[line_no]
                    key = (line_no, start_col, end_col)
                    if key in mapper.mappings:
                        mapper.mappings[key] = []  # Mark as deleted
                
                # Remove the line
                del lines[idx]
                
                # Update all mappings after this line (shift up by 1)
                mapper.update_all_mappings_after_line(line_no, -1)
                mapper.update_line_offset(-1)
                        
        elif action == 'replace':
            replace_start, replace_end = fix.get('replace_range', (idx, idx + 1))
            replacement_lines = fix['lines']
            
            # Calculate which original lines are being replaced
            original_lines_count = replace_end - replace_start
            replacement_count = len(replacement_lines)
            
            # Update mappings for original lines being replaced
            for i in range(original_lines_count):
                original_line_num = replace_start + 1 + i  # +1 because replace_start is 0-indexed
                if original_line_num in main_lines:
                    start_col, end_col = main_lines[original_line_num]
                    key = (original_line_num, start_col, end_col)
                    
                    if key in mapper.mappings:
                        # Calculate which fixed lines this original line maps to
                        current_line = mapper.get_adjusted_line(original_line_num)
                        if i < replacement_count:
                            # Maps to replacement line(s)
                            if original_lines_count == 1 and replacement_count > 1:
                                # 1 original line -> multiple replacement lines
                                fixed_lines = list(range(current_line, current_line + replacement_count))
                            else:
                                # 1:1 or many:many mapping
                                fixed_lines = [current_line + i] if i < replacement_count else []
                            
                            mapper.mappings[key] = fixed_lines
                        else:
                            # More original lines than replacement lines - mark as deleted
                            mapper.mappings[key] = []
            
            # Replace the lines
            lines[replace_start:replace_end] = replacement_lines
            
            # Update all mappings after the replacement
            lines_diff = replacement_count - original_lines_count
            if lines_diff != 0:
                mapper.update_all_mappings_after_line(replace_end, lines_diff)
                mapper.update_line_offset(lines_diff)
                        
        elif action == 'insert':
            insert_lines = fix['lines']
            lines[idx:idx] = insert_lines
            
            lines_added = len(insert_lines)
            # Update all mappings after the insertion point
            mapper.update_all_mappings_after_line(line_no, lines_added)
            mapper.update_line_offset(lines_added)
    
    return lines

def auto_fix_with_mapping(target_path: Path, debug: bool = False) -> Tuple[bool, Optional[Dict]]:
    """
    Auto-fix a Python file and return success status with complete line mappings.
    
    Args:
        target_path: Path to the Python file to fix
        debug: Whether to print debug information
        
    Returns:
        Tuple of (success: bool, mappings: Dict or None)
        - success: True if file was successfully fixed and runs without errors
        - mappings: Dictionary with structure {(original_line, start_col, end_col): [fixed_lines]}
                   None if the process failed
    """
    max_iterations = 15
    iteration = 0
    
    # Initialize the line mapper
    mapper = LineMapper()
    # Extract main function lines from original file
    original_main_lines = extract_main_function_lines(target_path)
    
    # Initialize mappings for ALL main function lines (1:1 mapping initially)
    for line_no, (start_col, end_col) in original_main_lines.items():
        mapper.add_mapping(line_no, start_col, end_col, [line_no])
    
    if debug:
        print(f"🚀 Starting auto-fix process for: {target_path}")
        print(f"📊 Found {len(original_main_lines)} lines in main function")
        print(f"📋 Initialized {len(mapper.mappings)} line mappings")
        print("=" * 60)
    
    while iteration < max_iterations:
        iteration += 1
        if debug: print(f"Iteration {iteration}...")
        
        # Re-extract main lines if they were empty due to syntax errors
        if not original_main_lines:
            original_main_lines = extract_main_function_lines(target_path)
            if debug and original_main_lines:
                print(f"📊 Extracted {len(original_main_lines)} main function lines after syntax fix")
        
        # First, check for syntax errors before trying to execute
        has_syntax_error, syntax_error_info = check_for_syntax_errors(target_path)
        
        if has_syntax_error:
            if debug: print(f"🔍 Syntax error detected before execution!")
            err_type, err_msg, line_no = syntax_error_info
            if debug: print(f"Detected {err_type} at line {line_no}: {err_msg}")
            
            # Handle syntax error with mapping
            lines = target_path.read_text(encoding="utf-8").splitlines()
            lines = handle_syntax_error_with_mapping(lines, line_no, err_msg, mapper, original_main_lines)
            target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            
            if debug: print(f"✅ Applied syntax error fix at line {line_no}")
            if debug: print("🔄 Retrying...\n")
            continue
        
        # If no syntax errors, try to execute the file
        if debug: print("🏃 Executing file to check for runtime errors...")
        proc = subprocess.run([sys.executable, str(target_path)], capture_output=True, text=True)

        if proc.returncode == 0:
            if debug: 
                print(f"\n🎉 SUCCESS! {target_path.name} runs without errors after {iteration} iterations!")
                print(f"📈 Total line mappings created: {len(mapper.mappings)}")
                print("=" * 60)
            return True, mapper.mappings

        # Check if this might be a syntax error that wasn't caught by compile()
        if 'SyntaxError' in proc.stderr:
            if debug: print("🔍 Syntax error detected in execution output!")
            syntax_parsed = parse_syntax_error(proc.stderr, target_path)
            if syntax_parsed:
                err_type, err_msg, line_no = syntax_parsed
                if debug: print(f"Detected {err_type} at line {line_no}: {err_msg}")
                
                # Handle syntax error with mapping
                lines = target_path.read_text(encoding="utf-8").splitlines()
                lines = handle_syntax_error_with_mapping(lines, line_no, err_msg, mapper, original_main_lines)
                target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                
                if debug: print(f"✅ Applied syntax error fix at line {line_no}")
                if debug: print("🔄 Retrying...\n")
                continue

        # Check for import errors and stop if needed
        if 'ImportError' in proc.stderr or 'ModuleNotFoundError' in proc.stderr:
            if debug: print("🚫 Import error detected - checking if required libraries are missing...")
            
            # Parse the import error to get the module name
            import_error_match = re.search(r"No module named '(\w+)'", proc.stderr)
            if import_error_match:
                missing_module = import_error_match.group(1)
                print(f"❌ ERROR: Required library '{missing_module}' is not installed.")
                print(f"📦 Please install it using: pip install {missing_module}")
                print("🛑 Stopping auto-fix process...")
                return False, None

        # If not a syntax error, use the regular error handling
        parsed = parse_traceback(proc.stderr, target_path)
        if not parsed:
            if debug: 
                print("❌ Could not parse error traceback; aborting.")
                print("STDERR:", proc.stderr)
            return False, None

        err_type, err_msg, line_no = parsed
        if debug: print(f"🐛 Detected {err_type} at line {line_no}: {err_msg}")

        lines = target_path.read_text(encoding="utf-8").splitlines()
        idx = (line_no - 1) if line_no and line_no > 0 else 0
        
        if idx >= len(lines):
            if debug: print("❌ Error line index out of range")
            return False, None
            
        line_text = lines[idx]
        if debug: print(f"🔍 Error line content: {line_text.strip()}")

        fix = make_fix(err_type, err_msg, line_text, lines, line_no)

        if isinstance(fix, dict):
            # Apply fix with mapping
            lines = apply_fix_with_mapping(lines, fix, line_no, mapper, original_main_lines)
            target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            
            if fix.get('action') in ['delete', 'smart_delete']:
                if debug: print(f"🗑️  Applied deletion at line {line_no}")
            else:
                if debug: print(f"🔧 Applied {fix['action']} at line {line_no}")
                if 'lines' in fix and debug:
                    print("   Fix content:")
                    for line in fix['lines']:
                        if debug: print(f"     + {line}")
        else:
            if debug: print("❌ No fix could be generated for this error")
            return False, None
            
        if debug: print("🔄 Retrying...\n")
    
    # If we reach here, we exceeded max iterations
    if debug: 
        print(f"⚠️  Maximum iterations ({max_iterations}) reached.")
        print("🔧 Manual intervention may be required.")
        print(f"📊 Partial mappings created: {len(mapper.mappings)}")
    
    return False, mapper.mappings  # Return partial mappings even on failure










































"""Aggiustare: se si trova un raise, sostituire l'intera riga cin un pass"""

import time

if __name__ == "__main__":
    file_path = os.path.abspath("C:/Users/rical/OneDrive/Desktop/QSmell_Tool/qspire/generated_executables/executable_initializer.py")
    success, mappings = auto_fix_with_mapping(Path(file_path), debug=True)

    print("\nLine Mappings:")
    for mapping in mappings:
        print(mapping,end='\t')
        print(" -> ", end="\t")
        print(mappings[mapping])



"""
My python project is about fixing another python file that has a main in order to prevent execution errors. This works fine. What I need to do, however, is to keep a map of the lines of the main before and after the fixes.

At the end of the auto fix function I need to have teh mapping of every line of the main before the fixes to the lines after the fix.

I should have everything, so I will not send you the whole functions, since they are a lot.

Inside the auto_fix function I have all the logic I need. The only problem is that I have a subfunction called make_fix, which says what is the fix to be done. Particular attention has the replace action of the make fix. The replace is special because it could even replace more lines. That's why I need a raplace_range, that tells how man

"""