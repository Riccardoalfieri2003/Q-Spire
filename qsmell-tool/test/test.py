import re
from typing import Dict, Set

def debug_function_body_detection(source_code: str, function_name: str):
    """Debug function to see what's happening with body detection."""
    
    def _get_all_comment_and_docstring_lines(source_code: str) -> Set[int]:
        """Simplified version for debugging."""
        comment_lines = set()
        lines = source_code.split('\n')
        
        in_triple_quote = False
        triple_quote_type = None
        
        for line_num, line in enumerate(lines, 1):
            original_line = line
            stripped_line = line.strip()
            
            if in_triple_quote:
                comment_lines.add(line_num)
                if triple_quote_type in original_line:
                    count = original_line.count(triple_quote_type)
                    if count % 2 == 1:
                        in_triple_quote = False
                        triple_quote_type = None
                continue
            
            if not stripped_line:
                continue
            
            if stripped_line.startswith('#'):
                comment_lines.add(line_num)
                continue
            
            patterns = [('"""', '"""'), ("'''", "'''")]
            triple_quote_found = False
            for pattern, quote_type in patterns:
                if pattern in original_line:
                    comment_lines.add(line_num)
                    triple_quote_found = True
                    
                    pattern_pos = original_line.find(pattern)
                    remaining_line = original_line[pattern_pos + len(pattern):]
                    
                    if quote_type not in remaining_line:
                        in_triple_quote = True
                        triple_quote_type = quote_type
                    break
            
            if triple_quote_found:
                continue
                
            if '#' in original_line:
                hash_pos = original_line.find('#')
                before_hash = original_line[:hash_pos]
                single_quotes = before_hash.count("'")
                double_quotes = before_hash.count('"')
                if single_quotes % 2 == 0 and double_quotes % 2 == 0:
                    comment_lines.add(line_num)
        
        return comment_lines

    def _find_function_end_regex(lines, start_index):
        """Find function end."""
        if start_index >= len(lines):
            return start_index
        
        def_line = lines[start_index]
        base_indent = len(def_line) - len(def_line.lstrip())
        
        for i in range(start_index + 1, len(lines)):
            line = lines[i]
            if not line.strip():
                continue
            
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent:
                stripped = line.strip()
                if (stripped.startswith('def ') or 
                    stripped.startswith('class ') or
                    stripped.startswith('if ') or
                    stripped.startswith('for ') or
                    stripped.startswith('while ') or
                    stripped.startswith('try:') or
                    stripped.startswith('with ') or
                    not stripped.startswith('#')):
                    return i
        
        return len(lines)

    # Find all functions
    lines = source_code.split('\n')
    comment_lines = _get_all_comment_and_docstring_lines(source_code)
    
    func_pattern = rf'^\s*def\s+{re.escape(function_name)}\s*\('
    
    functions_found = []
    
    for line_num, line in enumerate(lines, 1):
        if re.match(func_pattern, line):
            print(f"\n=== FOUND FUNCTION {len(functions_found)+1} at line {line_num} ===")
            print(f"Definition: {line.strip()}")
            
            start_line = line_num
            end_line = _find_function_end_regex(lines, line_num - 1)
            
            print(f"Function spans lines {start_line} to {end_line}")
            
            # Extract lines
            function_lines = []
            for func_line_num in range(start_line, min(end_line + 1, len(lines) + 1)):
                if func_line_num <= len(lines):
                    line_content = lines[func_line_num - 1]
                    
                    if (line_content.strip() and func_line_num not in comment_lines):
                        function_lines.append({
                            "line_number": func_line_num,
                            "content": line_content,
                            "stripped_content": line_content.strip(),
                        })
            
            print(f"Non-comment lines found: {len(function_lines)}")
            
            # Show first few lines for debugging
            for i, line_info in enumerate(function_lines[:10]):  # Show first 10 lines
                print(f"  Line {line_info['line_number']}: {repr(line_info['stripped_content'])}")
            
            if len(function_lines) > 10:
                print(f"  ... and {len(function_lines) - 10} more lines")
            
            # Test body detection
            function_dict = {
                "name": function_name,
                "start_line": start_line,
                "end_line": end_line,
                "lines": function_lines
            }
            
            has_body = debug_has_function_body(function_dict)
            print(f"Has body: {has_body}")
            
            functions_found.append(function_dict)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total functions found: {len(functions_found)}")
    functions_with_body = [f for f in functions_found if debug_has_function_body(f)]
    print(f"Functions with body: {len(functions_with_body)}")


def debug_has_function_body(function_dict: Dict) -> bool:
    """Debug version that shows what's being filtered."""
    if not function_dict.get("lines"):
        print("  No lines found")
        return False
    
    print(f"  Checking {len(function_dict['lines'])} lines for meaningful content:")
    
    for i, line_info in enumerate(function_dict["lines"]):
        stripped_content = line_info["stripped_content"]
        
        # Skip empty lines
        if not stripped_content:
            print(f"    Line {i+1}: SKIP (empty)")
            continue
            
        # Skip function definition line
        if stripped_content.startswith("def "):
            print(f"    Line {i+1}: SKIP (function definition) - {repr(stripped_content)}")
            continue
            
        # Skip function parameter lines
        if (stripped_content.endswith(",") or 
            stripped_content.endswith("(") or 
            (":" in stripped_content and "->" not in stripped_content and not stripped_content.endswith(":"))):
            
            if (re.match(r'^\s*\w+\s*:', stripped_content) or
                re.match(r'^\s*\w+\s*:\s*\w+.*,\s*$', stripped_content) or
                stripped_content in ["self,", "cls,"] or
                stripped_content.startswith("self,") or
                stripped_content.startswith("cls,") or
                re.match(r'^\s*\w+\s*:\s*\w+.*[,)]?\s*$', stripped_content)):
                print(f"    Line {i+1}: SKIP (parameter) - {repr(stripped_content)}")
                continue
        
        # Skip function signature continuation
        if (stripped_content.endswith("):") or 
            stripped_content.endswith(") ->") or
            "-> " in stripped_content):
            print(f"    Line {i+1}: SKIP (signature continuation) - {repr(stripped_content)}")
            continue
            
        # Skip stub patterns
        if (stripped_content == "pass" or
            stripped_content == "..." or
            stripped_content.endswith(": ...") or
            stripped_content.startswith("pass  #") or
            stripped_content.startswith("...  #") or
            stripped_content == "return" or
            stripped_content == "return None" or
            "NotImplementedError" in stripped_content or
            "raise NotImplementedError" in stripped_content.lower() or
            (stripped_content.startswith('"""') and stripped_content.endswith('"""')) or
            (stripped_content.startswith("'''") and stripped_content.endswith("'''"))):
            print(f"    Line {i+1}: SKIP (stub pattern) - {repr(stripped_content)}")
            continue
        
        # Found meaningful content!
        print(f"    Line {i+1}: MEANINGFUL! - {repr(stripped_content)}")
        return True
    
    print("  No meaningful lines found")
    return False


# Usage:
# debug_function_body_detection(source_code, 'to_other_device')


# Read your file
with open(r"C:\Users\rical\OneDrive\Desktop\QSmell_Tool\qsmell-tool\mpqp\mpqp\core\circuit.py", 'r', encoding='utf-8', errors='ignore') as f:
    source_code = f.read()

# Debug the function detection
debug_function_body_detection(source_code, 'to_other_device')