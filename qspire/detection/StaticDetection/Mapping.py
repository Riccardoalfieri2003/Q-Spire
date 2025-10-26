import ast
import difflib
from typing import List, Dict, Tuple, Optional, Any
import re

def map_lines_of_code(original_file: str, original_function: str, generated_file: str, similarity_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Map lines of code from an original function to the generated main function.
    
    Args:
        original_file: Path to the original Python file
        original_function: Name of the function in the original file
        generated_file: Path to the generated Python file (contains main function)
        similarity_threshold: Minimum similarity ratio (0.0 to 1.0) for line matching
        
    Returns:
        Dictionary containing the best mapping found with metadata
    """
    try:
        # Read both files
        with open(original_file, 'r', encoding='utf-8') as f:
            original_code = f.read()
        
        with open(generated_file, 'r', encoding='utf-8') as f:
            generated_code = f.read()
        
        # Parse AST for both files
        original_ast = ast.parse(original_code)
        generated_ast = ast.parse(generated_code)
        
        # Find all functions with the target name in original file
        original_functions = _find_functions_by_name(original_ast, original_function, original_code)
        
        # Find main block in generated file
        main_block = _find_main_block(generated_ast, generated_code)
        
        if not original_functions:
            return {
                "success": False,
                "error": f"Function '{original_function}' not found in {original_file}",
                "mapping": None
            }
        
        if not main_block:
            return {
                "success": False,
                "error": f"'if __name__ == \"__main__\":' block not found in {generated_file}",
                "mapping": None
            }
        
        # Try mapping each original function and find the best one
        best_mapping = None
        best_score = -1
        
        for i, orig_func in enumerate(original_functions):
            mapping = _map_function_lines(
                orig_func, 
                main_block, 
                similarity_threshold
            )
            
            # Calculate score (number of non-None mappings)
            score = sum(1 for m in mapping["line_mappings"] if m["generated_line"] is not None)
            
            if score > best_score:
                best_score = score
                best_mapping = mapping
                best_mapping["original_function_index"] = i
        
        return {
            "success": True,
            "mapping": best_mapping,
            "total_original_functions": len(original_functions),
            "best_match_score": best_score
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing files: {str(e)}",
            "mapping": None
        }


def _find_functions_by_name(ast_tree: ast.AST, function_name: str, source_code: str) -> List[Dict]:
    """Find all functions with the given name in the AST."""
    functions = []
    source_lines = source_code.split('\n')
    
    # Get all comment and docstring lines to filter them out
    comment_lines = _get_all_comment_and_docstring_lines(source_code)
    
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Extract function lines
            start_line = node.lineno - 1  # Convert to 0-based indexing
            end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1
            
            # If end_lineno is not available, estimate it
            if not hasattr(node, 'end_lineno'):
                end_line = _estimate_function_end(ast_tree, node, source_lines)
            
            function_lines = []
            for line_num in range(start_line, min(end_line, len(source_lines))):
                line_content = source_lines[line_num]
                line_number_1_based = line_num + 1
                
                # Skip empty lines and all types of comments/docstrings
                if (line_content.strip() and 
                    line_number_1_based not in comment_lines):
                    function_lines.append({
                        "line_number": line_number_1_based,
                        "content": line_content,
                        "stripped_content": line_content.strip(),
                        "start_col": len(line_content) - len(line_content.lstrip()),
                        "end_col": len(line_content)
                    })
            
            functions.append({
                "ast_node": node,
                "start_line": start_line + 1,
                "end_line": end_line,
                "lines": function_lines
            })
    
    return functions


def _find_main_block(ast_tree: ast.AST, source_code: str) -> Optional[Dict]:
    """Find the 'if __name__ == \"__main__\":' block in the AST."""
    source_lines = source_code.split('\n')
    
    # Get all comment and docstring lines to filter them out
    comment_lines = _get_all_comment_and_docstring_lines(source_code)
    
    for node in ast.walk(ast_tree):
        if isinstance(node, ast.If):
            # Check if this is the main block condition
            if _is_main_block_condition(node.test):
                start_line = node.lineno - 1  # Convert to 0-based indexing
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1
                
                # If end_lineno is not available, estimate it
                if not hasattr(node, 'end_lineno'):
                    end_line = _estimate_block_end(ast_tree, node, source_lines)
                
                # Extract the lines inside the if block (not including the if statement itself)
                block_lines = []
                
                # Start from the line after the if statement
                block_start = start_line + 1
                
                for line_num in range(block_start, min(end_line, len(source_lines))):
                    line_content = source_lines[line_num]
                    line_number_1_based = line_num + 1
                    
                    # Skip empty lines and all types of comments/docstrings
                    if (line_content.strip() and 
                        line_number_1_based not in comment_lines):
                        block_lines.append({
                            "line_number": line_number_1_based,
                            "content": line_content,
                            "stripped_content": line_content.strip(),
                            "start_col": len(line_content) - len(line_content.lstrip()) if line_content.strip() else 0,
                            "end_col": len(line_content)
                        })
                
                return {
                    "ast_node": node,
                    "start_line": start_line + 1,  # Line of the if statement
                    "end_line": end_line,
                    "block_start_line": block_start + 1,  # First line inside the block
                    "lines": block_lines
                }
    
    return None


def _is_main_block_condition(test_node) -> bool:
    """Check if an AST node represents the __name__ == '__main__' condition."""
    # Handle: __name__ == '__main__' or '__main__' == __name__
    if isinstance(test_node, ast.Compare):
        # Check if we have exactly one comparator
        if len(test_node.ops) == 1 and len(test_node.comparators) == 1:
            op = test_node.ops[0]
            left = test_node.left
            right = test_node.comparators[0]
            
            # Must be an equality comparison
            if isinstance(op, ast.Eq):
                # Check both possible orders: __name__ == '__main__' and '__main__' == __name__
                return (
                    (_is_name_node(left) and _is_main_string(right)) or
                    (_is_main_string(left) and _is_name_node(right))
                )
    
    return False


def _is_name_node(node) -> bool:
    """Check if node represents __name__."""
    return (isinstance(node, ast.Name) and node.id == '__name__')


def _is_main_string(node) -> bool:
    """Check if node represents '__main__' string."""
    if isinstance(node, ast.Constant):
        return node.value == '__main__'
    elif isinstance(node, ast.Str):  # Python < 3.8 compatibility
        return node.s == '__main__'
    return False


def _estimate_block_end(ast_tree: ast.AST, target_node: ast.If, source_lines: List[str]) -> int:
    """Estimate the end line of an if block when end_lineno is not available."""
    # Find all nodes that come after this if block
    all_nodes = []
    for node in ast.walk(ast_tree):
        if (hasattr(node, 'lineno') and 
            node.lineno > target_node.lineno and 
            node != target_node):
            all_nodes.append(node)
    
    if all_nodes:
        # Find the closest node that starts after our block
        next_line = min(node.lineno for node in all_nodes)
        return next_line - 1
    else:
        # If no nodes after, assume block goes to end of file
        return len(source_lines)


def _estimate_function_end(ast_tree: ast.AST, target_node: ast.FunctionDef, source_lines: List[str]) -> int:
    """Estimate the end line of a function when end_lineno is not available."""
    # Find all nodes that come after this function
    all_nodes = []
    for node in ast.walk(ast_tree):
        if hasattr(node, 'lineno') and node.lineno > target_node.lineno:
            all_nodes.append(node)
    
    if all_nodes:
        # Find the closest node that starts after our function
        next_line = min(node.lineno for node in all_nodes)
        return next_line - 1
    else:
        # If no nodes after, assume function goes to end of file
        return len(source_lines)


def _map_function_lines(original_func: Dict, main_block: Dict, similarity_threshold: float) -> Dict:
    """Map lines from original function to main block."""
    original_lines = original_func["lines"]
    main_lines = main_block["lines"]
    
    mappings = []
    main_search_start = 0
    
    for orig_line in original_lines:
        best_match = None
        best_similarity = 0
        
        # Search from the last found position onwards
        for i in range(main_search_start, len(main_lines)):
            main_line = main_lines[i]
            similarity = _calculate_line_similarity(
                orig_line["stripped_content"], 
                main_line["stripped_content"]
            )
            
            if similarity >= similarity_threshold and similarity > best_similarity:
                best_match = {
                    "generated_line": main_line["line_number"],
                    "generated_content": main_line["content"],
                    "generated_start_col": main_line["start_col"],
                    "generated_end_col": main_line["end_col"],
                    "similarity": similarity,
                    "main_line_index": i
                }
                best_similarity = similarity
        
        if best_match:
            # Update search start position for next iteration
            main_search_start = best_match["main_line_index"] + 1
            mappings.append({
                "original_line": orig_line["line_number"],
                "original_content": orig_line["content"],
                "original_start_col": orig_line["start_col"],
                "original_end_col": orig_line["end_col"],
                "generated_line": best_match["generated_line"],
                "generated_content": best_match["generated_content"],
                "generated_start_col": best_match["generated_start_col"],
                "generated_end_col": best_match["generated_end_col"],
                "similarity": best_match["similarity"]
            })
        else:
            # No match found
            mappings.append({
                "original_line": orig_line["line_number"],
                "original_content": orig_line["content"],
                "original_start_col": orig_line["start_col"],
                "original_end_col": orig_line["end_col"],
                "generated_line": None,
                "generated_content": None,
                "generated_start_col": None,
                "generated_end_col": None,
                "similarity": 0.0
            })
    
    return {
        "original_function": {
            "start_line": original_func["start_line"],
            "end_line": original_func["end_line"],
            "total_lines": len(original_lines)
        },
        "generated_main_block": {
            "if_statement_line": main_block["start_line"],
            "block_start_line": main_block["block_start_line"],
            "end_line": main_block["end_line"],
            "total_lines": len(main_lines)
        },
        "line_mappings": mappings,
        "successful_mappings": sum(1 for m in mappings if m["generated_line"] is not None),
        "total_mappings": len(mappings)
    }


def _get_all_comment_and_docstring_lines(source_code: str) -> set:
    """
    Get all line numbers that contain comments or docstrings using tokenization.
    This properly handles multiline comments, docstrings, and inline comments.
    """
    import tokenize
    import io
    
    comment_lines = set()
    
    try:
        # Use tokenize to properly identify all comments and strings
        tokens = tokenize.generate_tokens(io.StringIO(source_code).readline)
        
        for token in tokens:
            if token.type == tokenize.COMMENT:
                # Add all lines covered by this comment
                start_line, end_line = token.start[0], token.end[0]
                for line_num in range(start_line, end_line + 1):
                    comment_lines.add(line_num)
            
            elif token.type == tokenize.STRING:
                # Check if this string is likely a docstring
                if _is_likely_docstring(token, source_code):
                    # Add all lines covered by this docstring
                    start_line, end_line = token.start[0], token.end[0]
                    for line_num in range(start_line, end_line + 1):
                        comment_lines.add(line_num)
    
    except tokenize.TokenError:
        # Fallback to simple line-by-line analysis if tokenization fails
        lines = source_code.split('\n')
        in_multiline_string = False
        multiline_delimiter = None
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Handle multiline strings
            if in_multiline_string:
                comment_lines.add(i)
                if multiline_delimiter in line:
                    in_multiline_string = False
                    multiline_delimiter = None
            elif stripped.startswith('"""') or stripped.startswith("'''"):
                comment_lines.add(i)
                delimiter = '"""' if stripped.startswith('"""') else "'''"
                # Check if it's a single-line docstring
                if stripped.count(delimiter) < 2:
                    in_multiline_string = True
                    multiline_delimiter = delimiter
            elif stripped.startswith('#'):
                comment_lines.add(i)
    
    return comment_lines


def _is_likely_docstring(token, source_code: str) -> bool:
    """
    Determine if a string token is likely a docstring.
    This is a heuristic based on position and content.
    """
    # Get the line content
    lines = source_code.split('\n')
    token_line = token.start[0]
    
    if token_line <= len(lines):
        line_content = lines[token_line - 1].strip()
        
        # If the string starts at the beginning of a line (after whitespace)
        # and uses triple quotes, it's likely a docstring
        if (line_content.startswith('"""') or line_content.startswith("'''") or
            line_content.startswith('r"""') or line_content.startswith("r'''")):
            return True
        
        # Also check if it's the first statement after a function/class definition
        if token_line > 1:
            prev_line = lines[token_line - 2].strip()
            if (prev_line.endswith(':') and 
                (prev_line.startswith('def ') or prev_line.startswith('class ') or
                 'def ' in prev_line or 'class ' in prev_line)):
                return True
    
    return False


def _is_comment_only_line(line: str) -> bool:
    """Check if a line contains only comments (and whitespace)."""
    stripped = line.strip()
    if not stripped:
        return False  # Empty lines are handled separately
    
    # Check if line starts with # (comment)
    if stripped.startswith('#'):
        return True
    
    # Check for docstrings (triple quotes)
    if (stripped.startswith('"""') or stripped.startswith("'''") or
        stripped.startswith('r"""') or stripped.startswith("r'''")):
        return True
    
    # Check if it's a continuation or end of a multiline docstring
    if (stripped.endswith('"""') or stripped.endswith("'''")) and len(stripped) >= 3:
        # If it only contains the closing quotes, it's a docstring line
        content_without_quotes = stripped.replace('"""', '').replace("'''", '').strip()
        if not content_without_quotes:
            return True
    
    return False


def _calculate_line_similarity(line1: str, line2: str) -> float:
    """Calculate similarity between two lines of code."""
    if not line1 and not line2:
        return 1.0
    if not line1 or not line2:
        return 0.0
    
    # Remove extra whitespace and normalize
    line1_normalized = re.sub(r'\s+', ' ', line1.strip())
    line2_normalized = re.sub(r'\s+', ' ', line2.strip())
    
    # Use difflib to calculate similarity
    similarity = difflib.SequenceMatcher(None, line1_normalized, line2_normalized).ratio()
    
    return similarity


# Example usage function
def print_mapping_results(result: Dict):
    """Helper function to print mapping results in a readable format."""
    if not result["success"]:
        print(f"Error: {result['error']}")
        return
    
    mapping = result["mapping"]
    print(f"Mapping Results:")
    print(f"- Total original functions found: {result['total_original_functions']}")
    print(f"- Best match score: {result['best_match_score']}")
    print(f"- Successful mappings: {mapping['successful_mappings']}/{mapping['total_mappings']}")
    print(f"- Success rate: {mapping['successful_mappings']/mapping['total_mappings']*100:.1f}%")
    
    print(f"\nOriginal function: lines {mapping['original_function']['start_line']}-{mapping['original_function']['end_line']}")
    print(f"Generated main block: if statement at line {mapping['generated_main_block']['if_statement_line']}")
    print(f"Generated main block content: lines {mapping['generated_main_block']['block_start_line']}-{mapping['generated_main_block']['end_line']}")
    
    print(f"\nLine-by-line mapping:")
    for i, line_map in enumerate(mapping['line_mappings']):
        if line_map['generated_line'] is not None:
            print(f"Original {line_map['original_line']:3d} -> Generated {line_map['generated_line']:3d} (similarity: {line_map['similarity']:.2f})")
        else:
            print(f"Original {line_map['original_line']:3d} -> No match found")


















def extract_line_mappings(result: Dict) -> Dict[int, Dict]:
    """
    Extract line number mappings with column information from both original and generated code.
    
    Args:
        result: The result dictionary returned by map_lines_of_code()
        
    Returns:
        Dictionary mapping original line numbers to mapping info with columns for both sides.
        Lines that couldn't be mapped are excluded from the result.
        
    Example:
        {
            15: {
                "original": {
                    "line": 15,
                    "start_col": 0,
                    "end_col": 20
                },
                "generated": {
                    "line": 52,
                    "start_col": 4,
                    "end_col": 25
                }
            }
        }
    """
    line_mappings = {}
    
    # Check if the result was successful
    if not result.get("success", False) or not result.get("mapping"):
        return line_mappings
    
    # Extract line mappings from the result
    mapping_data = result["mapping"]["line_mappings"]
    
    for line_map in mapping_data:
        original_line = line_map["original_line"]
        generated_line = line_map["generated_line"]
        
        # Only include mappings where a corresponding line was found
        if generated_line is not None:
            line_mappings[original_line] = {
                "original": {
                    "line": line_map["original_line"],
                    "start_col": line_map["original_start_col"],
                    "end_col": line_map["original_end_col"]
                },
                "generated": {
                    "line": line_map["generated_line"],
                    "start_col": line_map["generated_start_col"],
                    "end_col": line_map["generated_end_col"]
                }
            }
    
    return line_mappings


def extract_line_mappings_with_none(result: Dict) -> Dict[int, Optional[Dict]]:
    """
    Extract line number mappings with column information, including None values for unmapped lines.
    
    Args:
        result: The result dictionary returned by map_lines_of_code()
        
    Returns:
        Dictionary mapping original line numbers to mapping info with columns for both sides.
        Lines that couldn't be mapped have None as their value.
        
    Example:
        {
            15: {
                "original": {
                    "line": 15,
                    "start_col": 0,
                    "end_col": 20
                },
                "generated": {
                    "line": 52,
                    "start_col": 4,
                    "end_col": 25
                }
            },
            17: None,  # Original line 17 couldn't be mapped
            18: {
                "original": {
                    "line": 18,
                    "start_col": 4,
                    "end_col": 15
                },
                "generated": {
                    "line": 55,
                    "start_col": 0,
                    "end_col": 12
                }
            }
        }
    """
    line_mappings = {}
    
    # Check if the result was successful
    if not result.get("success", False) or not result.get("mapping"):
        return line_mappings
    
    # Extract line mappings from the result
    mapping_data = result["mapping"]["line_mappings"]
    
    for line_map in mapping_data:
        original_line = line_map["original_line"]
        generated_line = line_map["generated_line"]
        
        # Include all mappings, with None for failed ones
        if generated_line is not None:
            line_mappings[original_line] = {
                "original": {
                    "line": line_map["original_line"],
                    "start_col": line_map["original_start_col"],
                    "end_col": line_map["original_end_col"]
                },
                "generated": {
                    "line": line_map["generated_line"],
                    "start_col": line_map["generated_start_col"],
                    "end_col": line_map["generated_end_col"]
                }
            }
        else:
            line_mappings[original_line] = None
    
    return line_mappings


def extract_simple_line_mappings(result: Dict) -> Dict[int, int]:
    """
    Extract simple line-to-line mappings (just line numbers, no columns).
    
    Args:
        result: The result dictionary returned by map_lines_of_code()
        
    Returns:
        Dictionary mapping original line numbers to generated line numbers.
        Lines that couldn't be mapped are excluded from the result.
        
    Example:
        {
            15: 52,  # Original line 15 maps to generated line 52
            16: 53,  # Original line 16 maps to generated line 53
            18: 55   # Original line 18 maps to generated line 55
        }
    """
    simple_mappings = {}
    
    # Check if the result was successful
    if not result.get("success", False) or not result.get("mapping"):
        return simple_mappings
    
    # Extract line mappings from the result
    mapping_data = result["mapping"]["line_mappings"]
    
    for line_map in mapping_data:
        original_line = line_map["original_line"]
        generated_line = line_map["generated_line"]
        
        # Only include mappings where a corresponding line was found
        if generated_line is not None:
            simple_mappings[original_line] = generated_line
    
    return simple_mappings


def extract_complete_mappings(result: Dict) -> Dict[int, Dict]:
    """
    Extract complete mapping information including both original and generated line details.
    
    Args:
        result: The result dictionary returned by map_lines_of_code()
        
    Returns:
        Dictionary with complete mapping information for each original line.
        Only includes successful mappings.
        
    Example:
        {
            15: {
                "original": {
                    "line": 15,
                    "start_col": 0,
                    "end_col": 20,
                    "content": "    x = 5"
                },
                "generated": {
                    "line": 52,
                    "start_col": 4,
                    "end_col": 25,
                    "content": "    x = 5"
                },
                "similarity": 1.0
            }
        }
    """
    complete_mappings = {}
    
    # Check if the result was successful
    if not result.get("success", False) or not result.get("mapping"):
        return complete_mappings
    
    # Extract line mappings from the result
    mapping_data = result["mapping"]["line_mappings"]
    
    for line_map in mapping_data:
        original_line = line_map["original_line"]
        generated_line = line_map["generated_line"]
        
        # Only include successful mappings
        if generated_line is not None:
            complete_mappings[original_line] = {
                "original": {
                    "line": line_map["original_line"],
                    "start_col": line_map["original_start_col"],
                    "end_col": line_map["original_end_col"],
                    "content": line_map["original_content"]
                },
                "generated": {
                    "line": line_map["generated_line"],
                    "start_col": line_map["generated_start_col"],
                    "end_col": line_map["generated_end_col"],
                    "content": line_map["generated_content"]
                },
                "similarity": line_map["similarity"]
            }
    
    return complete_mappings









#Example usage:
result = map_lines_of_code(
     "C:/Users/rical/OneDrive/Desktop/QSmell_Tool/qspire/mpqp/mpqp/core/circuit.py", 
     "initializer", 
     "C:/Users/rical/OneDrive/Desktop/QSmell_Tool/qspire/generated_executables/executable_initializer.py",
     similarity_threshold=0.4
 )
print_mapping_results(result)



print()


simple_result=extract_line_mappings(result)
for line in simple_result:
    print(simple_result[line])



"""

{'C:\\Users\\rical\\OneDrive\\Desktop\\QSmell_Tool\\qspire\\generated_executables\\executable_initializer.py': 'C:\\Users\\rical\\OneDrive\\Desktop\\QSmell_Tool\\qspire\\mpqp\\mpqp\\core\\circuit.py',
 'C:\\Users\\rical\\OneDrive\\Desktop\\QSmell_Tool\\qspire\\generated_executables\\executable_to_other_language.py': 'C:\\Users\\rical\\OneDrive\\Desktop\\QSmell_Tool\\qspire\\mpqp\\mpqp\\core\\circuit.py',
 'C:\\Users\\rical\\OneDrive\\Desktop\\QSmell_Tool\\qspire\\generated_executables\\executable_to_other_device.py': 'C:\\Users\\rical\\OneDrive\\Desktop\\QSmell_Tool\\qspire\\mpqp\\mpqp\\core\\circuit.py',
 'C:\\Users\\rical\\OneDrive\\Desktop\\QSmell_Tool\\qspire\\generated_executables\\executable_from_other_language.py': 'C:\\Users\\rical\\OneDrive\\Desktop\\QSmell_Tool\\qspire\\mpqp\\mpqp\\core\\circuit.py'}
"""