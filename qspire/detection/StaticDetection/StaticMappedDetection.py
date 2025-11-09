import csv
from datetime import datetime
from pathlib import Path
import shutil
import threading
import os
from detection.StaticDetection.StaticCircuit import FunctionExecutionGenerator
#from test.GeneralFolderTest import save_output

















"""
    This section is used to map the classes and functions of the generated file to the original one
"""

from typing import Any, Optional

def map_lines_simple(original_file: str, generated_file: str, similarity_threshold: float = 0.6) -> dict[str, Any]:
    """
    Simple line-by-line mapping between two files based on similarity.
    
    Args:
        original_file: Path to the original Python file
        generated_file: Path to the generated Python file  
        similarity_threshold: Minimum similarity ratio (0.0 to 1.0) for line matching
        
    Returns:
        dictionary containing mapping results
    """
    try:
        # Read both files
        with open(original_file, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()
        
        with open(generated_file, 'r', encoding='utf-8') as f:
            generated_lines = f.readlines()
        
        # Identify commented lines in both files to skip them
        generated_comment_lines = _get_comment_lines(''.join(generated_lines))
        original_comment_lines = _get_comment_lines(''.join(original_lines))
        
        # Find the main block start line in generated file
        main_block_start = _find_main_block_start(generated_lines)

        
        # Create mappings for each generated line
        mappings = []
        
        for gen_line_num, gen_line in enumerate(generated_lines, 1):

            if gen_line_num in generated_comment_lines: continue
            if gen_line_num >= main_block_start: break

            gen_line_clean = gen_line.strip()
            
            # Skip completely empty lines
            if not gen_line_clean:
                mappings.append({
                    "generated_line": gen_line_num,
                    "generated_content": gen_line.rstrip('\n'),
                    "original_line": None,
                    "original_content": None,
                    "similarity": 0.0
                })
                continue
            
            best_match_line = None
            best_similarity = 0.0
            best_original_content = None
            
            # Compare with every line in original file (skip comments and empty lines)
            for orig_line_num, orig_line in enumerate(original_lines, 1):
                # Skip commented lines in original file
                if orig_line_num in original_comment_lines:
                    continue
                    
                orig_line_clean = orig_line.strip()
                
                # Skip empty original lines
                if not orig_line_clean:
                    continue
                
                # Calculate similarity
                similarity = _calculate_line_similarity(gen_line_clean, orig_line_clean)
                
                if similarity >= similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_line = orig_line_num
                    best_original_content = orig_line.rstrip('\n')
            
            # Add mapping result
            mappings.append({
                "generated_line": gen_line_num,
                "generated_content": gen_line.rstrip('\n'),
                "original_line": best_match_line,
                "original_content": best_original_content,
                "similarity": best_similarity
            })
        
        # Calculate statistics
        successful_mappings = sum(1 for m in mappings if m["original_line"] is not None)
        
        return {
            "success": True,
            "original_file": original_file,
            "generated_file": generated_file,
            "total_generated_lines": len(generated_lines),
            "total_original_lines": len(original_lines),
            "successful_mappings": successful_mappings,
            "total_mappings": len(mappings),
            "success_rate": successful_mappings / len(mappings) if mappings else 0.0,
            "similarity_threshold": similarity_threshold,
            "line_mappings": mappings
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error processing files: {str(e)}"
        }


def _find_main_block_start(lines: list) -> Optional[int]:
    """
    Find the line number where the main block starts (if __name__ == "__main__":).
    Returns None if no main block is found.
    """
    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()
        # Check for main block pattern
        if (stripped_line.startswith('if __name__') and '__main__' in stripped_line and 
            stripped_line.endswith(':')):
            return line_num
    return None


def _get_comment_lines(source_code: str) -> set[int]:
    """
    Get all line numbers that contain comments or docstrings using regex patterns.
    This approach is more robust and doesn't rely on Python tokenization.
    Works even with malformed/incomplete Python code.
    """
    comment_lines = set()
    lines = source_code.split('\n')
    
    in_triple_quote = False
    triple_quote_type = None
    
    for line_num, line in enumerate(lines, 1):
        original_line = line
        line = line.strip()
        
        # Handle continuation of triple-quoted strings
        if in_triple_quote:
            comment_lines.add(line_num)
            # Check if triple quote ends on this line
            if triple_quote_type in original_line:
                # Count occurrences to handle edge cases
                count = original_line.count(triple_quote_type)
                if count % 2 == 1:  # Odd number means it closes
                    in_triple_quote = False
                    triple_quote_type = None
            continue
        
        # Skip empty lines
        if not line:
            continue
            
        # Check for single-line comments (lines starting with #)
        if line.startswith('#'):
            comment_lines.add(line_num)
            continue
        
        # Check for triple-quoted strings (docstrings)
        triple_quote_patterns = [
            (r'"""', '"""'),
            (r"'''", "'''"),
            (r'r"""', '"""'),
            (r"r'''", "'''"),
            (r'u"""', '"""'),
            (r"u'''", "'''"),
        ]
        
        found_triple_quote = False
        for pattern, quote_type in triple_quote_patterns:
            if pattern in original_line.lower():
                comment_lines.add(line_num)
                found_triple_quote = True
                
                # Check if it's a single-line docstring
                first_occurrence = original_line.lower().find(pattern)
                remaining_line = original_line[first_occurrence + len(pattern):]
                
                if quote_type in remaining_line:
                    # Single-line docstring, already handled
                    pass
                else:
                    # Multi-line docstring starts
                    in_triple_quote = True
                    triple_quote_type = quote_type
                break
        
        if found_triple_quote:
            continue
            
        # Check for inline comments (# somewhere in the line)
        # Simple heuristic: if # appears and it's likely not in a string
        if '#' in original_line:
            # Very basic check: count quotes before the #
            hash_index = original_line.find('#')
            before_hash = original_line[:hash_index]
            
            # Count single and double quotes
            single_quotes = before_hash.count("'")
            double_quotes = before_hash.count('"')
            
            # If even number of quotes, # is likely a comment
            if single_quotes % 2 == 0 and double_quotes % 2 == 0:
                comment_lines.add(line_num)
    
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


def _calculate_line_similarity(line1: str, line2: str) -> float:
    """Calculate similarity between two lines of code."""
    if not line1 and not line2:
        return 1.0
    if not line1 or not line2:
        return 0.0
    
    # Remove extra whitespace and normalize
    import re
    line1_normalized = re.sub(r'\s+', ' ', line1.strip())
    line2_normalized = re.sub(r'\s+', ' ', line2.strip())
    
    # Use difflib to calculate similarity
    import difflib
    similarity = difflib.SequenceMatcher(None, line1_normalized, line2_normalized).ratio()
    
    return similarity


# Helper functions to extract different types of mappings

def get_simple_line_mappings(result: dict) -> dict[int, int]:
    """
    Get simple generated_line -> original_line mappings.
    Only includes successful mappings.
    """
    if not result.get("success", False):
        return {}
    
    simple_mappings = {}
    for mapping in result["line_mappings"]:
        if mapping["original_line"] is not None:
            simple_mappings[mapping["generated_line"]] = mapping["original_line"]
    
    return simple_mappings


def get_detailed_line_mappings(result: dict) -> dict[int, dict]:
    """
    Get detailed mappings with content and similarity info.
    Only includes successful mappings.
    """
    if not result.get("success", False):
        return {}
    
    detailed_mappings = {}
    for mapping in result["line_mappings"]:
        if mapping["original_line"] is not None:
            detailed_mappings[mapping["generated_line"]] = {
                "original_line": mapping["original_line"],
                "original_content": mapping["original_content"],
                "generated_content": mapping["generated_content"],
                "similarity": mapping["similarity"]
            }
    
    return detailed_mappings


def get_all_mappings_with_none(result: dict) -> dict[int, Optional[int]]:
    """
    Get all mappings including None for unmapped lines.
    """
    if not result.get("success", False):
        return {}
    
    all_mappings = {}
    for mapping in result["line_mappings"]:
        all_mappings[mapping["generated_line"]] = mapping["original_line"]
    
    return all_mappings


def print_mapping_summary(result: dict):
    """Print a summary of the mapping results."""
    if not result.get("success", False):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    print(f"Line Mapping Summary:")
    print(f"  Original file: {result['original_file']}")
    print(f"  Generated file: {result['generated_file']}")
    print(f"  Total generated lines: {result['total_generated_lines']}")
    print(f"  Total original lines: {result['total_original_lines']}")
    print(f"  Successfully mapped: {result['successful_mappings']}/{result['total_mappings']}")
    print(f"  Success rate: {result['success_rate']:.1%}")
    print(f"  Similarity threshold: {result['similarity_threshold']}")


def print_detailed_mappings(result: dict, max_lines: int = 20):
    """Print detailed line-by-line mappings."""
    if not result.get("success", False):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    print(f"\nDetailed Line Mappings (showing first {max_lines} lines):")
    print("-" * 80)
    
    for i, mapping in enumerate(result["line_mappings"][:max_lines]):
        gen_line = mapping["generated_line"]
        orig_line = mapping["original_line"]
        similarity = mapping["similarity"]
        
        if orig_line is not None:
            print(f"Generated {gen_line:3d} -> Original {orig_line:3d} (similarity: {similarity:.2f})")
        else:
            print(f"Generated {gen_line:3d} -> No match found")
    
    if len(result["line_mappings"]) > max_lines:
        print(f"... and {len(result['line_mappings']) - max_lines} more lines")





























































"""
    This section maps the liens inside the generated main with the function of the original file
"""

import ast
import difflib
from typing import List, Dict, Tuple, Optional, Any
import re



def _map_function_lines_regex(original_func: Dict, main_block: Dict, similarity_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Map lines from main block to original function using similarity matching.
    For each line in the main block, find the best matching line in the original function.
    """
    from difflib import SequenceMatcher
    import re
    
    def normalize_line(line: str) -> str:
        """Normalize a line for comparison by removing extra whitespace and comments."""
        # Remove comments
        line = re.sub(r'#.*$', '', line)
        # Remove extra whitespace
        line = ' '.join(line.split())
        # Remove common Python keywords that might differ
        line = re.sub(r'\bself\.\b', '', line)
        # Remove common variations
        line = re.sub(r'\bmain\(\)', '', line)  # Remove main() calls
        return line.strip().lower()
    
    def calculate_similarity(line1: str, line2: str) -> float:
        """Calculate similarity between two lines."""
        norm1 = normalize_line(line1)
        norm2 = normalize_line(line2)
        
        if not norm1 or not norm2:
            return 0.0
            
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    original_lines = original_func['lines']
    main_lines = main_block['lines']
    
    line_mappings = []
    
    # For each line in the MAIN BLOCK, find the best match in the ORIGINAL FUNCTION
    for main_line in main_lines:
        main_content = main_line['stripped_content']
        
        # Skip empty lines, comments, and obvious non-code lines
        if (not main_content or 
            main_content.startswith('#') or 
            main_content.startswith('try:') or
            main_content.startswith('except') or
            main_content.startswith('"""') or
            main_content.startswith("'''")):
            line_mappings.append({
                "generated_line": main_line['line_number'],
                "generated_content": main_line['content'],
                "original_line": None,
                "original_content": None,
                "similarity": 0.0
            })
            continue
        
        best_match = None
        best_similarity = 0.0
        
        # Find the best matching line in original function
        for orig_line in original_lines:
            orig_content = orig_line['stripped_content']
            
            # Skip empty lines, comments, and function definition
            if (not orig_content or 
                orig_content.startswith('#') or 
                orig_content.startswith('def ') or
                orig_content.startswith('"""') or
                orig_content.startswith("'''")):
                continue
                
            similarity = calculate_similarity(main_content, orig_content)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = orig_line
        
        # Only include the mapping if similarity meets threshold
        if best_match and best_similarity >= similarity_threshold:
            line_mappings.append({
                "generated_line": main_line['line_number'],
                "generated_content": main_line['content'],
                "original_line": best_match['line_number'],
                "original_content": best_match['content'],
                "similarity": best_similarity
            })
        else:
            line_mappings.append({
                "generated_line": main_line['line_number'],
                "generated_content": main_line['content'],
                "original_line": None,
                "original_content": None,
                "similarity": best_similarity if best_match else 0.0
            })
    
    return {
        "line_mappings": line_mappings,
        "total_main_lines": len(main_lines),
        "total_mapped_lines": sum(1 for m in line_mappings if m["original_line"] is not None)
    }


# Updated main mapping function
def map_lines_of_code(original_file: str, original_function: str, generated_file: str, similarity_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Map lines of code from main block to original function.
    For each line in the main block, find the best matching line in the original function.
    
    Args:
        original_file: Path to the original Python file
        original_function: Name of the function in the original file
        generated_file: Path to the generated Python file (contains main function)
        similarity_threshold: Minimum similarity ratio (0.0 to 1.0) for line matching
        
    Returns:
        Dictionary containing the mapping from main block lines to original function lines
    """
    try:
        # Read both files
        with open(original_file, 'r', encoding='utf-8', errors='ignore') as f:
            original_code = f.read()
        
        with open(generated_file, 'r', encoding='utf-8', errors='ignore') as f:
            generated_code = f.read()
        
        # Find all functions with the target name in original file
        all_functions = _find_functions_by_name_regex(original_code, original_function)
        
        # Filter to only keep functions with actual body
        original_functions = [func for func in all_functions if _has_function_body(func)]
        
        # Find main block in generated file
        main_block = _find_main_block_regex(generated_code)
        
        if not original_functions:
            return {
                "success": False,
                "error": f"Function '{original_function}' not found in {original_file} (found {len(all_functions)} function definitions but none with actual body)",
                "mapping": None
            }
        
        if not main_block:
            return {
                "success": False,
                "error": f"'if __name__ == \"__main__\":' block not found in {generated_file}",
                "mapping": None
            }
        
        # Try mapping with each original function and find the best one
        best_mapping = None
        best_score = -1
        
        for i, orig_func in enumerate(original_functions):

            #if orig_func['name']=="construct_circuit":

            """
            print(f"Found {len(original_functions)} functions named '{original_function}' with actual body")
            print(f"Main block starts at line {main_block['start_line']} with {len(main_block['lines'])} lines")

            print("FUNCTION")
            print(orig_func['lines'])

            print("MAIN")
            print(main_block)

            print(f"\n=== Trying function {i+1}/{len(original_functions)}: {orig_func.get('name', original_function)} ===")
            
            # Debug output for the construct_circuit function
            if orig_func.get('name') == original_function or original_function in str(orig_func.get('lines', [{}])[0].get('content', '')):
                print("FUNCTION DETAILS:")
                print(f"  Start line: {orig_func.get('start_line', 'unknown')}")
                print(f"  Total lines: {len(orig_func['lines'])}")
                print("  First 10 lines:")
                for j, line in enumerate(orig_func['lines'][:10]):
                    print(f"    {line['line_number']}: '{line['stripped_content']}'")
                
                print("\nMAIN BLOCK DETAILS:")
                print(f"  Start line: {main_block['start_line']}")
                print(f"  Total lines: {len(main_block['lines'])}")
                print("  First 10 lines:")
                for j, line in enumerate(main_block['lines'][:10]):
                    print(f"    {line['line_number']}: '{line['stripped_content']}'")
            """
            
            mapping = _map_function_lines_regex(
                orig_func, 
                main_block, 
                similarity_threshold
            )
            
            #print(f"Mapping result: {mapping['total_mapped_lines']}/{mapping['total_main_lines']} executable lines mapped")
            #print(f"Original function had {mapping['total_executable_original_lines']} executable lines")
            
            # Calculate score (number of successful mappings)
            score = mapping['total_mapped_lines']
            
            if score > best_score:
                best_score = score
                best_mapping = mapping
                best_mapping["original_function_index"] = i
            
            """
            # Print some example mappings for debugging
            print("Sample mappings:")
            successful_mappings = [m for m in mapping["line_mappings"] if m["original_line"] is not None]
            failed_mappings = [m for m in mapping["line_mappings"] if m["original_line"] is None]
            
            # Show successful mappings
            
            for j, line_map in enumerate(successful_mappings[:3]):
                main_content = line_map['generated_content'].strip()
                orig_content = line_map['original_content'].strip()
                print(f"  ✓ Main {line_map['generated_line']}: '{main_content[:50]}{'...' if len(main_content) > 50 else ''}' -> "
                    f"Original {line_map['original_line']}: '{orig_content[:50]}{'...' if len(orig_content) > 50 else ''}' "
                    f"(similarity: {line_map['similarity']:.3f})")
            
            # Show failed mappings
            for j, line_map in enumerate(failed_mappings[:2]):
                main_content = line_map['generated_content'].strip()
                print(f"  ✗ Main {line_map['generated_line']}: '{main_content[:50]}{'...' if len(main_content) > 50 else ''}' -> "
                    f"No match ({line_map.get('reason', 'unknown reason')})")"""
        
        if not best_mapping:
            return {
                "success": False,
                "error": "No successful mappings found between any original function and main block",
                "mapping": None
            }
        
        """print(f"\n=== BEST MAPPING SELECTED ===")
        print(f"Function index: {best_mapping['original_function_index']}")
        print(f"Total mapped lines: {best_mapping['total_mapped_lines']}")
        print(f"Success rate: {best_mapping['total_mapped_lines']}/{best_mapping['total_main_lines']} = {(best_mapping['total_mapped_lines']/max(1, best_mapping['total_main_lines']))*100:.1f}%")"""
        
        return {
            "success": True,
            "mapping": best_mapping,
            "total_original_functions": len(original_functions),
            "best_match_score": best_score
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Error processing files: {str(e)}",
            "traceback": traceback.format_exc(),
            "mapping": None
        }


def _map_function_lines_regex(original_func: Dict, main_block: Dict, similarity_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Map lines from main block to original function using similarity matching.
    For each line in the main block, find the best matching line in the original function.
    IMPORTANTLY: Skip function definition lines and focus on actual code content.
    """
    from difflib import SequenceMatcher
    import re
    
    def normalize_line(line: str) -> str:
        """Normalize a line for comparison by removing extra whitespace and comments."""
        # Remove comments
        line = re.sub(r'#.*$', '', line)
        # Remove extra whitespace
        line = ' '.join(line.split())
        # Remove common Python keywords that might differ
        line = re.sub(r'\bself\.\b', '', line)
        # Remove common variations
        line = re.sub(r'\bmain\(\)', '', line)
        return line.strip().lower()
    
    def is_function_header_line(content: str) -> bool:
        """Check if a line is part of the function definition (header)."""
        content = content.strip()
        return (content.startswith('def ') or 
                (content.startswith(')') and content.endswith(':')) or
                ('-> ' in content and content.endswith(':')) or
                (content.count('(') > 0 and content.count(')') == 0 and not '=' in content))  # Parameter lines
    
    def is_executable_code(content: str) -> bool:
        """Check if a line contains actual executable code."""
        content = content.strip()
        if not content:
            return False
        
        # Skip obvious non-executable lines
        skip_patterns = [
            content.startswith('#'),           # Comments
            content.startswith('"""'),         # Docstrings
            content.startswith("'''"),         # Docstrings
            content.startswith('try:'),        # Try blocks (usually error handling in main)
            content.startswith('except'),      # Exception handling
            content == 'pass',                 # Pass statements
            content == '...',                  # Ellipsis
            is_function_header_line(content)   # Function headers
        ]
        
        return not any(skip_patterns)
    
    def calculate_similarity(line1: str, line2: str) -> float:
        """Calculate similarity between two lines."""
        norm1 = normalize_line(line1)
        norm2 = normalize_line(line2)
        
        if not norm1 or not norm2:
            return 0.0
            
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    original_lines = original_func['lines']
    main_lines = main_block['lines']
    
    # Filter original lines to only include executable code
    executable_original_lines = [
        line for line in original_lines 
        if is_executable_code(line['stripped_content'])
    ]
    
    # Filter main lines to only include executable code
    executable_main_lines = [
        line for line in main_lines
        if is_executable_code(line['stripped_content'])
    ]
    
    #print(f"  Original function: {len(original_lines)} total lines, {len(executable_original_lines)} executable")
    #print(f"  Main block: {len(main_lines)} total lines, {len(executable_main_lines)} executable")
    
    line_mappings = []
    
    # For each line in the MAIN BLOCK, find the best match in the ORIGINAL FUNCTION
    for main_line in main_lines:
        main_content = main_line['stripped_content']
        
        # Skip non-executable lines in main block
        if not is_executable_code(main_content):
            line_mappings.append({
                "generated_line": main_line['line_number'],
                "generated_content": main_line['content'],
                "original_line": None,
                "original_content": None,
                "similarity": 0.0,
                "reason": "Non-executable main line"
            })
            continue
        
        best_match = None
        best_similarity = 0.0
        
        # Find the best matching line in executable original lines
        for orig_line in executable_original_lines:
            similarity = calculate_similarity(main_content, orig_line['stripped_content'])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = orig_line
        
        # Only include the mapping if similarity meets threshold
        if best_match and best_similarity >= similarity_threshold:
            line_mappings.append({
                "generated_line": main_line['line_number'],
                "generated_content": main_line['content'],
                "original_line": best_match['line_number'],
                "original_content": best_match['content'],
                "similarity": best_similarity,
                "reason": "Successful match"
            })
        else:
            line_mappings.append({
                "generated_line": main_line['line_number'],
                "generated_content": main_line['content'],
                "original_line": None,
                "original_content": None,
                "similarity": best_similarity if best_match else 0.0,
                "reason": f"Below threshold ({best_similarity:.3f} < {similarity_threshold})" if best_match else "No similar line found"
            })
    
    return {
        "line_mappings": line_mappings,
        "total_main_lines": len(executable_main_lines),
        "total_mapped_lines": sum(1 for m in line_mappings if m["original_line"] is not None),
        "total_executable_original_lines": len(executable_original_lines)
    }

def _has_function_body(function_dict: Dict) -> bool:
    """
    Check if a function has actual body content (not just annotations or empty).
    
    Args:
        function_dict: Dictionary containing function information with 'lines' key
        
    Returns:
        True if function has meaningful body content, False otherwise
    """
    if not function_dict.get("lines"):
        return False
    
    # Check all lines in the function for meaningful content
    for line_info in function_dict["lines"]:
        stripped_content = line_info["stripped_content"]
        original_content = line_info["content"]
        
        # Skip empty lines
        if not stripped_content:
            continue
            
        # Skip function definition line (def function_name(...)
        if stripped_content.startswith("def "):
            continue
            
        # Skip function parameter lines (lines that are part of function signature)
        if (stripped_content.endswith(",") or 
            stripped_content.endswith("(") or 
            (":" in stripped_content and "->" not in stripped_content and not stripped_content.endswith(":"))):
            # This might be a parameter line, but let's be more specific
            # Skip if it looks like a parameter (has type annotation pattern)
            if (re.match(r'^\s*\w+\s*:', stripped_content) or  # param: Type
                re.match(r'^\s*\w+\s*:\s*\w+.*,\s*$', stripped_content) or  # param: Type,
                stripped_content in ["self,", "cls,"] or  # self or cls parameters
                stripped_content.startswith("self,") or
                stripped_content.startswith("cls,") or
                # Match parameter patterns like "device: ATOSDevice," or "translation_warning: bool = True,"
                re.match(r'^\s*\w+\s*:\s*\w+.*[,)]?\s*$', stripped_content)):
                continue
        
        # Skip lines that are just function signature continuation
        if (stripped_content.endswith("):") or 
            stripped_content.endswith(") ->") or
            "-> " in stripped_content):
            continue
            
        # Skip common stub patterns
        if (stripped_content == "pass" or
            stripped_content == "..." or
            stripped_content.endswith(": ...") or  # function signature ending with : ...
            stripped_content.startswith("pass  #") or
            stripped_content.startswith("...  #") or
            stripped_content == "return" or
            stripped_content == "return None" or
            # Skip raise NotImplementedError (common in stubs)
            "NotImplementedError" in stripped_content or
            "raise NotImplementedError" in stripped_content.lower() or
            # Skip just docstrings at function level
            (stripped_content.startswith('"""') and stripped_content.endswith('"""')) or
            (stripped_content.startswith("'''") and stripped_content.endswith("'''"))):
            continue
        
        # If we reach here, we found a meaningful line
        return True
    
    # No meaningful lines found
    return False


def _find_functions_by_name_regex(source_code: str, function_name: str) -> List[Dict]:
    """
    Find all functions with the given name using regex patterns.
    Works even with malformed Python code.
    """
    functions = []
    lines = source_code.split('\n')
    
    # Get all comment and docstring lines to filter them out
    comment_lines = _get_all_comment_and_docstring_lines(source_code)
    
    # Pattern to match function definition
    func_pattern = rf'^\s*def\s+{re.escape(function_name)}\s*\('
    
    for line_num, line in enumerate(lines, 1):
        if re.match(func_pattern, line):
            # Found function definition
            start_line = line_num
            
            # Find function end by looking for next function/class or end of indentation
            end_line = _find_function_end_regex(lines, line_num - 1)  # Convert to 0-based
            
            # Extract non-comment, non-empty lines from the function
            function_lines = []
            for func_line_num in range(start_line, min(end_line + 1, len(lines) + 1)):
                if func_line_num <= len(lines):
                    line_content = lines[func_line_num - 1]  # Convert to 0-based for lines list
                    
                    # Skip empty lines and all types of comments/docstrings
                    if (line_content.strip() and 
                        func_line_num not in comment_lines):
                        function_lines.append({
                            "line_number": func_line_num,
                            "content": line_content,
                            "stripped_content": line_content.strip(),
                            "start_col": len(line_content) - len(line_content.lstrip()),
                            "end_col": len(line_content)
                        })
            
            functions.append({
                "name": function_name,
                "start_line": start_line,
                "end_line": end_line,
                "lines": function_lines
            })
    
    return functions


def _find_function_end_regex(lines: List[str], start_index: int) -> int:
    """
    Find the end line of a function starting at start_index.
    Uses indentation-based detection and handles multi-line signatures.
    """
    if start_index >= len(lines):
        return start_index
    
    # Get the base indentation of the function definition
    def_line = lines[start_index]
    base_indent = len(def_line) - len(def_line.lstrip())
    
    # First, find where the function signature actually ends (look for the colon)
    signature_end = start_index
    found_colon = False
    
    for i in range(start_index, len(lines)):
        line = lines[i]
        stripped = line.strip()
        
        # Look for the colon that ends the function signature
        if ':' in stripped and not found_colon:
            # Make sure it's not inside quotes or comments
            colon_pos = stripped.find(':')
            before_colon = stripped[:colon_pos]
            
            # Simple check: if it ends with : and doesn't look like a type annotation
            if (stripped.endswith(':') or 
                stripped.endswith(': ...') or
                '-> ' in before_colon):  # Return type annotation before colon
                signature_end = i
                found_colon = True
                break
    
    # Now look for the actual end of the function body
    for i in range(signature_end + 1, len(lines)):
        line = lines[i]
        
        # Skip empty lines
        if not line.strip():
            continue
        
        current_indent = len(line) - len(line.lstrip())
        
        # If we find a line with same or less indentation than the function def
        if current_indent <= base_indent:
            stripped = line.strip()
            # Check if it's a new function, class, or other top-level construct
            if (stripped.startswith('def ') or 
                stripped.startswith('class ') or
                stripped.startswith('@') or  # decorator
                (not stripped.startswith('#') and stripped)):  # Not a comment and not empty
                return i  # Return 1-based line number
    
    # If we reach here, function goes to end of file
    return len(lines)



def _find_main_block_regex(source_code: str) -> Optional[Dict]:
    """
    Find the main block using regex patterns.
    Takes all lines from the if __name__ == "__main__": line to the end of the file.
    """
    lines = source_code.split('\n')
    comment_lines = _get_all_comment_and_docstring_lines(source_code)
    
    # Pattern to match if __name__ == "__main__":
    main_pattern = r'^\s*if\s+__name__\s*==\s*["\']__main__["\']?\s*:\s*$'
    
    for line_num, line in enumerate(lines, 1):
        if re.match(main_pattern, line):
            # Found main block start
            start_line = line_num + 1  # Start from the line after the if statement
            
            # Extract all non-comment lines from the main block to end of file
            main_lines = []
            
            for main_line_num in range(start_line, len(lines) + 1):
                if main_line_num <= len(lines):
                    line_content = lines[main_line_num - 1]  # Convert to 0-based
                    
                    # Skip empty lines
                    if not line_content.strip():
                        continue
                    
                    # Skip comments and docstrings
                    if main_line_num not in comment_lines:
                        main_lines.append({
                            "line_number": main_line_num,
                            "content": line_content,
                            "stripped_content": line_content.strip(),
                            "start_col": len(line_content) - len(line_content.lstrip()),
                            "end_col": len(line_content)
                        })
            
            return {
                "start_line": start_line,
                "lines": main_lines
            }
    
    return None

"""
def _map_function_lines_regex(original_function: Dict, main_block: Dict, similarity_threshold: float) -> Dict:

    import difflib
    
    mappings = []
    
    for orig_line in original_function["lines"]:
        orig_content = orig_line["stripped_content"]
        
        best_match = None
        best_similarity = 0.0
        
        # Compare with each line in main block
        for main_line in main_block["lines"]:
            main_content = main_line["stripped_content"]
            
            # Calculate similarity
            similarity = difflib.SequenceMatcher(None, orig_content, main_content).ratio()
            
            if similarity >= similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = main_line
        
        # Add mapping result
        mappings.append({
            "original_line": orig_line["line_number"],
            "original_content": orig_line["content"].rstrip('\n'),
            "generated_line": best_match["line_number"] if best_match else None,
            "generated_content": best_match["content"].rstrip('\n') if best_match else None,
            "similarity": best_similarity
        })
    
    # Calculate statistics
    successful_mappings = sum(1 for m in mappings if m["generated_line"] is not None)
    
    return {
        "original_function_start": original_function["start_line"],
        "original_function_end": original_function["end_line"],
        "main_block_start": main_block["start_line"],
        "total_original_lines": len(original_function["lines"]),
        "total_main_lines": len(main_block["lines"]),
        "successful_mappings": successful_mappings,
        "success_rate": successful_mappings / len(mappings) if mappings else 0.0,
        "similarity_threshold": similarity_threshold,
        "line_mappings": mappings
    }
"""

# Keep the comment detection function from the previous artifact
def _get_all_comment_and_docstring_lines(source_code: str) -> set[int]:
    """
    Get all line numbers that contain comments or docstrings using regex patterns.
    This approach is robust and doesn't rely on Python tokenization.
    Works even with malformed/incomplete Python code.
    """
    comment_lines = set()
    lines = source_code.split('\n')
    
    in_triple_quote = False
    triple_quote_type = None
    
    for line_num, line in enumerate(lines, 1):
        original_line = line
        stripped_line = line.strip()
        
        # Handle continuation of triple-quoted strings
        if in_triple_quote:
            comment_lines.add(line_num)
            # Check if triple quote ends on this line
            if triple_quote_type in original_line:
                count = original_line.count(triple_quote_type)
                if count % 2 == 1:  # Odd number means it closes
                    in_triple_quote = False
                    triple_quote_type = None
            continue
        
        # Skip empty lines
        if not stripped_line:
            continue
        
        # Check for single-line comments
        if stripped_line.startswith('#'):
            comment_lines.add(line_num)
            continue
        
        # Check for triple-quoted strings
        patterns = [('"""', '"""'), ("'''", "'''"), ('r"""', '"""'), ("r'''", "'''")]
        
        triple_quote_found = False
        for pattern, quote_type in patterns:
            if pattern.lower() in original_line.lower():
                comment_lines.add(line_num)
                triple_quote_found = True
                
                pattern_pos = original_line.lower().find(pattern.lower())
                remaining_line = original_line[pattern_pos + len(pattern):]
                
                if quote_type not in remaining_line:
                    in_triple_quote = True
                    triple_quote_type = quote_type
                break
        
        if triple_quote_found:
            continue
        
        # Check for inline comments
        if '#' in original_line:
            hash_pos = original_line.find('#')
            before_hash = original_line[:hash_pos]
            
            single_quotes = before_hash.count("'")
            double_quotes = before_hash.count('"')
            
            if single_quotes % 2 == 0 and double_quotes % 2 == 0:
                comment_lines.add(line_num)
    
    return comment_lines






def _find_functions_by_name(ast_tree: ast.AST, function_name: str, source_code: str) -> List[Dict]:
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

def _get_all_comment_and_docstring_lines(source_code: str) -> set[int]:
    """
    Get all line numbers that contain comments or docstrings using regex patterns.
    This approach is robust and doesn't rely on Python tokenization.
    Works even with malformed/incomplete Python code.
    
    Returns:
        Set of line numbers (1-based) that contain comments or docstrings
    """
    comment_lines = set()
    lines = source_code.split('\n')
    
    in_triple_quote = False
    triple_quote_type = None
    triple_quote_start_line = None
    
    for line_num, line in enumerate(lines, 1):
        original_line = line
        stripped_line = line.strip()
        
        # Handle continuation of triple-quoted strings
        if in_triple_quote:
            comment_lines.add(line_num)
            # Check if triple quote ends on this line
            if triple_quote_type in original_line:
                # Count occurrences to handle edge cases
                count = original_line.count(triple_quote_type)
                if count % 2 == 1:  # Odd number means it closes
                    in_triple_quote = False
                    triple_quote_type = None
                    triple_quote_start_line = None
            continue
        
        # Skip empty lines (don't mark them as comments)
        if not stripped_line:
            continue
        
        # Check for single-line comments (lines starting with #)
        if stripped_line.startswith('#'):
            comment_lines.add(line_num)
            continue
        
        # Check for triple-quoted strings (docstrings and multi-line strings)
        triple_quote_found = False
        
        # Look for different types of triple quotes
        patterns = [
            ('"""', '"""'),
            ("'''", "'''"),
        ]
        
        # Also check for raw strings and unicode strings
        raw_patterns = [
            ('r"""', '"""'),
            ("r'''", "'''"),
            ('u"""', '"""'),
            ("u'''", "'''"),
            ('b"""', '"""'),
            ("b'''", "'''"),
            ('f"""', '"""'),  # f-strings
            ("f'''", "'''"),
        ]
        
        all_patterns = patterns + raw_patterns
        
        for pattern, quote_type in all_patterns:
            pattern_lower = pattern.lower()
            original_lower = original_line.lower()
            
            if pattern_lower in original_lower:
                # Found start of triple quote
                comment_lines.add(line_num)
                triple_quote_found = True
                
                # Find the position of the pattern
                pattern_pos = original_lower.find(pattern_lower)
                remaining_line = original_line[pattern_pos + len(pattern):]
                
                # Check if it closes on the same line
                if quote_type in remaining_line:
                    # Count closing quotes in the remaining line
                    closing_count = remaining_line.count(quote_type)
                    if closing_count % 2 == 1:
                        # Single-line docstring/string, already handled by adding line_num
                        pass
                    else:
                        # Multi-line starts (even number means it doesn't close)
                        in_triple_quote = True
                        triple_quote_type = quote_type
                        triple_quote_start_line = line_num
                else:
                    # Multi-line docstring starts
                    in_triple_quote = True
                    triple_quote_type = quote_type
                    triple_quote_start_line = line_num
                break
        
        if triple_quote_found:
            continue
        
        # Check for inline comments (# somewhere in the line)
        if '#' in original_line:
            # Simple heuristic to avoid # inside strings
            hash_positions = []
            for i, char in enumerate(original_line):
                if char == '#':
                    hash_positions.append(i)
            
            for hash_pos in hash_positions:
                before_hash = original_line[:hash_pos]
                
                # Count quotes before the hash
                single_quotes = before_hash.count("'")
                double_quotes = before_hash.count('"')
                
                # Simple check: if even number of quotes, # is likely a comment
                # This is not perfect but works for most cases
                if single_quotes % 2 == 0 and double_quotes % 2 == 0:
                    # Additional check: make sure we're not inside a triple quote
                    if not _is_inside_multiline_string(before_hash):
                        comment_lines.add(line_num)
                        break
    
    return comment_lines

def _is_inside_multiline_string(text_before_hash: str) -> bool:
    """
    Simple heuristic to check if we might be inside a multiline string.
    This is not perfect but helps avoid false positives.
    """
    # Count triple quotes before the hash
    triple_double = text_before_hash.count('"""')
    triple_single = text_before_hash.count("'''")
    
    # If odd number of triple quotes, we might be inside a multiline string
    return (triple_double % 2 == 1) or (triple_single % 2 == 1)


def _find_main_block(ast_tree: ast.AST, source_code: str) -> Optional[Dict]:
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



def _get_simple_comment_lines(lines: list) -> set:
    """
    Simple fallback method to detect comment lines without using tokenize.
    This is more robust for files with syntax errors.
    """
    comment_lines = set()
    in_multiline_string = False
    multiline_delimiter = None
    
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
        
        # Handle multiline strings/docstrings
        if in_multiline_string:
            comment_lines.add(line_num)
            # Check if this line ends the multiline string
            if multiline_delimiter in line:
                # Count occurrences to handle cases like: '''text''' on same line
                delimiter_count = line.count(multiline_delimiter)
                if delimiter_count % 2 == 1:  # Odd number means it closes
                    in_multiline_string = False
                    multiline_delimiter = None
            continue
        
        # Check for start of multiline strings
        if ('"""' in stripped or "'''" in stripped):
            if stripped.startswith('"""') or stripped.startswith("'''"):
                delimiter = '"""' if stripped.startswith('"""') else "'''"
                comment_lines.add(line_num)
                # Check if it's a single-line docstring
                delimiter_count = stripped.count(delimiter)
                if delimiter_count == 1:  # Only opening delimiter
                    in_multiline_string = True
                    multiline_delimiter = delimiter
                # If count is 2 or more, it's a complete docstring on one line
                continue
        
        # Check for single-line comments
        if stripped.startswith('#'):
            comment_lines.add(line_num)
            continue
    
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



def _find_main_block_start(lines: list) -> Optional[int]:
    """
    Find the line number where the main block starts (if __name__ == "__main__":).
    Returns None if no main block is found.
    """
    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()
        # Check for main block pattern
        if (stripped_line.startswith('if __name__') and '__main__' in stripped_line and 
            stripped_line.endswith(':')):
            return line_num
    return None


def _calculate_line_similarity(line1: str, line2: str) -> float:
    """Calculate similarity between two lines of code."""
    if not line1 and not line2:
        return 1.0
    if not line1 or not line2:
        return 0.0
    
    # Remove extra whitespace and normalize
    import re
    line1_normalized = re.sub(r'\s+', ' ', line1.strip())
    line2_normalized = re.sub(r'\s+', ' ', line2.strip())
    
    # Use difflib to calculate similarity
    import difflib
    similarity = difflib.SequenceMatcher(None, line1_normalized, line2_normalized).ratio()
    
    return similarity



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


















































































































"""
    This section fixes the generated main and maps the fixes in respect to the initial generated main
"""


#!/usr/bin/env python3
import subprocess
import sys
import re
import os
from pathlib import Path



debug=False

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
                    """elif '.run(' in assignment:
                        return 'object'  # Job objects"""
                elif '.run(' in assignment:
                    return 'unknown'  # Job objects
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
                    return 'unkown'
                    #return 'object'  # Generic object for unknown cases
        return 'unkown'
        #return 'object'  # Default fallback
    
    def create_default_value(var_type):
        """Create appropriate default value based on inferred type"""
        defaults = {
            'str': '""',
            'int': '0',
            'float': '0.0',
            'bool': 'False',
            'list': '[]',
            'dict': '{}',
            'object': 'type("MockObject", (), {"__getattr__": lambda self, name: lambda *args, **kwargs: None, "__call__": lambda self, *args, **kwargs: None, "__iter__": lambda self: iter([]), "__len__": lambda self: 0, "__getitem__": lambda self, key: None, "__str__": lambda self: "", "__int__": lambda self: 0})()',
            #'unknown': 'type("MockObject", (), { "__getattr__": lambda self, name: lambda *args, **kwargs: None,  "__call__": lambda self, *args, **kwargs: None, "__iter__": lambda self: iter([]),  "__len__": lambda self: 0, "__getitem__": lambda self, key: None,  "__str__": lambda self: "",  "__int__": lambda self: 0 })()'

            'unknown': 'type("_Unknown", (), { "__getattr__": lambda self, name: self,"__call__": lambda self, *args, **kwargs: self,"__getitem__": lambda self, key: self,"__iter__": lambda self: iter([]), "__bool__": lambda self: True, "__str__": lambda self: "", "__repr__": lambda self: "_Unknown()", "__len__": lambda self: 0, "__eq__": lambda self, other: False, "__add__": lambda self, other: self, "__sub__": lambda self, other: self,"__mul__": lambda self, other: self,"__truediv__": lambda self, other: sel})()'

        }
        return defaults.get(var_type, defaults['unknown'])
        #return defaults.get(var_type, defaults['object'])
    
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
 


def get_type_for_attribute(missing_attr, is_callable):
    """
    Determine the appropriate type/value based on the missing attribute.
    Returns a tuple of (type_name, mock_value, import_statement)
    """
    
    # String methods
    string_methods = {
        'split', 'strip', 'replace', 'find', 'index', 'startswith', 'endswith',
        'upper', 'lower', 'capitalize', 'join', 'format', 'encode', 'decode',
        'isdigit', 'isalpha', 'isalnum', 'isspace', 'count', 'rfind', 'rindex',
        'lstrip', 'rstrip', 'center', 'ljust', 'rjust', 'zfill', 'expandtabs',
        'translate', 'maketrans', 'partition', 'rpartition', 'splitlines'
    }
    
    # List methods
    list_methods = {
        'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'index',
        'count', 'sort', 'reverse', 'copy'
    }
    
    # Dict methods
    dict_methods = {
        'keys', 'values', 'items', 'get', 'pop', 'popitem', 'clear',
        'update', 'setdefault', 'copy', 'fromkeys'
    }
    
    # File-like object methods
    file_methods = {
        'read', 'write', 'readline', 'readlines', 'writelines', 'seek',
        'tell', 'flush', 'close', 'fileno', 'isatty', 'readable', 'writable'
    }
    
    # Datetime methods
    datetime_methods = {
        'strftime', 'strptime', 'date', 'time', 'replace', 'isoformat',
        'weekday', 'isoweekday', 'isocalendar', 'timetuple', 'utctimetuple'
    }
    
    # Path-like methods (pathlib or os.path)
    path_methods = {
        'exists', 'is_file', 'is_dir', 'resolve', 'absolute', 'parent',
        'name', 'suffix', 'stem', 'parts', 'anchor', 'mkdir', 'rmdir',
        'unlink', 'rename', 'replace', 'chmod', 'stat', 'lstat'
    }
    
    # Regex match methods
    regex_methods = {
        'group', 'groups', 'groupdict', 'start', 'end', 'span'
    }
    
    # Common object attributes (non-callable)
    common_attributes = {
        'length': ('list', '[]'),
        'size': ('list', '[]'),
        'shape': ('tuple', '(0, 0)'),  # numpy-like
        'dtype': ('str', '"object"'),  # numpy-like
        'text': ('str', '""'),
        'content': ('str', '""'),
        'data': ('dict', '{}'),
        'value': ('int', '0'),
        'name': ('str', '""'),
        'id': ('int', '0'),
        'type': ('str', '""'),
        'status': ('str', '""'),
        'code': ('int', '0'),
        'message': ('str', '""'),
        'args': ('tuple', '()'),
        'kwargs': ('dict', '{}')
    }
    
    # Determine the appropriate type
    if missing_attr in string_methods:
        return 'str', '""', None
    elif missing_attr in list_methods:
        return 'list', '[]', None
    elif missing_attr in dict_methods:
        return 'dict', '{}', None
    elif missing_attr in file_methods:
        return 'io.StringIO', 'io.StringIO()', 'import io'
    elif missing_attr in datetime_methods:
        return 'datetime.datetime', 'datetime.datetime.now()', 'import datetime'
    elif missing_attr in path_methods:
        return 'pathlib.Path', 'pathlib.Path(".")', 'import pathlib'
    elif missing_attr in regex_methods:
        return 'Mock', 'Mock()', 'from unittest.mock import Mock'
    elif missing_attr in common_attributes:
        attr_type, attr_value = common_attributes[missing_attr]
        return attr_type, attr_value, None
    else:
        # Fallback: create a mock object that can handle any attribute/method call
        if is_callable:
            return 'Mock', 'Mock()', 'from unittest.mock import Mock'
        else:
            # For unknown non-callable attributes, try to infer from name
            if missing_attr.endswith('_count') or missing_attr.endswith('_num'):
                return 'int', '0', None
            elif missing_attr.endswith('_name') or missing_attr.endswith('_text'):
                return 'str', '""', None
            elif missing_attr.endswith('_list') or missing_attr.endswith('_items'):
                return 'list', '[]', None
            elif missing_attr.endswith('_dict') or missing_attr.endswith('_map'):
                return 'dict', '{}', None
            else:
                return 'Mock', 'Mock()', 'from unittest.mock import Mock'


def get_smart_mock_value(missing_attr, is_callable):
    """
    Get appropriate mock value based on the missing attribute name.
    """
    # String methods
    string_methods = {
        'split', 'strip', 'replace', 'find', 'index', 'startswith', 'endswith',
        'upper', 'lower', 'capitalize', 'join', 'format', 'encode', 'decode',
        'isdigit', 'isalpha', 'isalnum', 'isspace', 'count', 'rfind', 'rindex'
    }
    
    # List methods
    list_methods = {
        'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'index',
        'count', 'sort', 'reverse', 'copy'
    }
    
    # Dict methods
    dict_methods = {
        'keys', 'values', 'items', 'get', 'pop', 'popitem', 'clear',
        'update', 'setdefault', 'copy', 'fromkeys'
    }
    
    # File-like object methods
    file_methods = {
        'read', 'write', 'readline', 'readlines', 'close', 'flush'
    }
    
    # Check if we need to redefine the variable type
    if missing_attr in string_methods:
        return 'REDEFINE_AS_STRING'
    elif missing_attr in list_methods:
        return 'REDEFINE_AS_LIST'
    elif missing_attr in dict_methods:
        return 'REDEFINE_AS_DICT'
    elif missing_attr in file_methods:
        return 'REDEFINE_AS_FILE'
    else:
        # Fallback to original logic
        return "lambda *args, **kwargs: None" if is_callable else "1"         



import inspect

def find_method_in_common_types(method_name):
    """
    Dynamically find which types have the given method by inspecting common types.
    Returns the most appropriate type and a sample instance.
    """
    
    # Common built-in types to check
    builtin_types = [
        (str, '""'),
        (list, '[]'),
        (dict, '{{}}'),
        (set, 'set()'),
        (tuple, '()'),
        (int, '0'),
        (float, '0.0'),
        (bytes, 'b""'),
    ]
    
    # Check built-in types first
    for type_class, sample_value in builtin_types:
        if hasattr(type_class, method_name):
            return type_class.__name__, sample_value, None
    
    # Check for numpy-like methods
    numpy_methods = {
        'astype', 'reshape', 'flatten', 'ravel', 'transpose', 'sum', 'mean', 
        'std', 'var', 'min', 'max', 'argmin', 'argmax', 'sort', 'argsort',
        'clip', 'round', 'abs', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tan'
    }
    
    if method_name in numpy_methods:
        # Try to import numpy and check if it's available
        return 'numpy.ndarray', 'numpy.array([])', 'import numpy'
    
    # Check for pandas-like methods
    pandas_methods = {
        'head', 'tail', 'info', 'describe', 'value_counts', 'groupby',
        'merge', 'join', 'concat', 'pivot', 'melt', 'drop', 'dropna',
        'fillna', 'isnull', 'notnull', 'apply', 'map', 'filter'
    }
    
    if method_name in pandas_methods:
        return 'pandas.DataFrame', 'pandas.DataFrame()', 'import pandas'
    
    # Check for datetime methods
    datetime_methods = {
        'strftime', 'strptime', 'date', 'time', 'replace', 'isoformat',
        'weekday', 'isoweekday', 'isocalendar', 'timetuple', 'utctimetuple'
    }
    
    if method_name in datetime_methods:
        return 'datetime.datetime', 'datetime.datetime.now()', 'import datetime'
    
    # Check for pathlib methods
    pathlib_methods = {
        'exists', 'is_file', 'is_dir', 'resolve', 'absolute', 'parent',
        'name', 'suffix', 'stem', 'parts', 'anchor', 'mkdir', 'rmdir',
        'unlink', 'rename', 'chmod', 'stat', 'lstat', 'glob', 'rglob'
    }
    
    if method_name in pathlib_methods:
        return 'pathlib.Path', 'pathlib.Path(".")', 'import pathlib'
    
    # Check for file-like methods
    file_methods = {
        'read', 'write', 'readline', 'readlines', 'writelines', 'seek',
        'tell', 'flush', 'close', 'fileno', 'readable', 'writable'
    }
    
    if method_name in file_methods:
        return 'io.StringIO', 'io.StringIO()', 'import io'
    
    # Check for requests/http methods
    requests_methods = {
        'json', 'text', 'content', 'status_code', 'headers', 'cookies',
        'raise_for_status', 'iter_content', 'iter_lines'
    }
    
    if method_name in requests_methods:
        return 'Mock', 'Mock()', 'from unittest.mock import Mock'
    
    # Advanced: Try to dynamically inspect available modules
    try:
        # Check if method exists in commonly imported modules
        common_modules = ['numpy', 'pandas', 'datetime', 'pathlib', 'io', 'json', 're']
        
        for module_name in common_modules:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                # Look for classes in the module that have this method
                for name in dir(module):
                    try:
                        obj = getattr(module, name)
                        if inspect.isclass(obj) and hasattr(obj, method_name):
                            # Found a class with this method
                            if module_name == 'numpy':
                                return 'numpy.ndarray', 'numpy.array([])', f'import {module_name}'
                            elif module_name == 'pandas':
                                return 'pandas.DataFrame', 'pandas.DataFrame()', f'import {module_name}'
                            else:
                                return f'{module_name}.{name}', f'{module_name}.{name}()', f'import {module_name}'
                    except:
                        continue
    except:
        pass
    
    # Final fallback - create a mock object
    return 'Mock', 'Mock()', 'from unittest.mock import Mock'







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

"""
def check_for_syntax_errors(target_path):
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

"""




















































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
"""
class LineMapper:
    
    def __init__(self):
        self.mappings = {}  # Dict with structure: {(original_line, start_col, end_col): [fixed_lines]}
        self.line_offset = 0  # Current offset for line numbers
    
    def add_mapping(self, original_line: int, original_start_col: int, original_end_col: int,
                   fixed_lines: List[int]):
        key = (original_line, original_start_col, original_end_col)
        self.mappings[key] = fixed_lines
    
    def add_deletion_mapping(self, original_line: int, original_start_col: int, original_end_col: int):
        key = (original_line, original_start_col, original_end_col)
        self.mappings[key] = []  # Empty list indicates deletion
    
    def update_line_offset(self, offset_change: int):
        self.line_offset += offset_change
    
    def get_adjusted_line(self, original_line: int) -> int:
        return original_line + self.line_offset
    
    def update_all_mappings_after_line(self, after_line: int, offset: int):
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

class LineMapper:
    """Tracks line number changes as code is modified during auto-fixing"""
    
    def __init__(self):
        self.mappings = {}  # Dict with structure: {(original_line, start_col, end_col): [fixed_lines]}
        self.cumulative_offset = 0  # Track total line changes
    
    def add_mapping(self, original_line: int, original_start_col: int, original_end_col: int,
                   fixed_lines: List[int]):
        """Add a mapping from original position to fixed lines"""
        key = (original_line, original_start_col, original_end_col)
        self.mappings[key] = fixed_lines
    
    def add_deletion_mapping(self, original_line: int, original_start_col: int, original_end_col: int):
        """Add a mapping for a deleted line"""
        key = (original_line, original_start_col, original_end_col)
        self.mappings[key] = []  # Empty list indicates deletion
        # Update offset for deletion
        self.cumulative_offset -= 1
    
    def add_insertion_mapping(self, original_line: int, original_start_col: int, original_end_col: int,
                             num_lines_inserted: int):
        """Add mapping for insertions and update offset"""
        current_fixed_line = original_line + self.cumulative_offset
        key = (original_line, original_start_col, original_end_col)
        
        # Create list of new line numbers
        new_lines = [current_fixed_line + i for i in range(num_lines_inserted)]
        self.mappings[key] = new_lines
        
        # Update offset for future mappings
        self.cumulative_offset += (num_lines_inserted - 1)  # -1 because original line is replaced
        
        # Update all existing mappings that come after this line
        self.update_mappings_after_change(current_fixed_line, num_lines_inserted - 1)
    
    def add_replacement_mapping(self, original_line: int, original_start_col: int, original_end_col: int,
                               num_replacement_lines: int):
        """Add mapping for line replacements"""
        current_fixed_line = original_line + self.cumulative_offset
        key = (original_line, original_start_col, original_end_col)
        
        # Create list of replacement line numbers
        new_lines = [current_fixed_line + i for i in range(num_replacement_lines)]
        self.mappings[key] = new_lines
        
        # Update offset (replacement of 1 line with N lines = +N-1 offset)
        offset_change = num_replacement_lines - 1
        self.cumulative_offset += offset_change
        
        # Update all existing mappings that come after this line
        if offset_change != 0:
            self.update_mappings_after_change(current_fixed_line, offset_change)
    
    def update_mappings_after_change(self, change_line: int, offset: int):
        """Update all existing mappings that come after a line change"""
        if offset == 0:
            return
            
        updated_mappings = {}
        for (orig_line, start_col, end_col), fixed_lines in self.mappings.items():
            if fixed_lines:  # Only update non-deleted mappings
                updated_fixed_lines = []
                for fixed_line in fixed_lines:
                    # Update lines that come after the change point
                    if fixed_line > change_line:
                        updated_fixed_lines.append(fixed_line + offset)
                    else:
                        updated_fixed_lines.append(fixed_line)
                updated_mappings[(orig_line, start_col, end_col)] = updated_fixed_lines
            else:
                # Keep deleted mappings unchanged
                updated_mappings[(orig_line, start_col, end_col)] = fixed_lines
        self.mappings = updated_mappings
    
    def get_current_line_number(self, original_line: int) -> int:
        """Get what line number an original line maps to currently"""
        return original_line + self.cumulative_offset
    
    def apply_fix_and_update_mapping(self, original_line: int, original_start_col: int, 
                                   original_end_col: int, fix_action: str, num_lines: int = 1):
        """Apply a fix and update mapping in one operation to ensure consistency"""
        if fix_action in ['delete', 'smart_delete']:
            self.add_deletion_mapping(original_line, original_start_col, original_end_col)
        elif fix_action == 'insert':
            self.add_insertion_mapping(original_line, original_start_col, original_end_col, num_lines)
        elif fix_action == 'replace':
            self.add_replacement_mapping(original_line, original_start_col, original_end_col, num_lines)
        else:
            # For other actions, assume 1:1 mapping with current offset
            current_line = self.get_current_line_number(original_line)
            self.add_mapping(original_line, original_start_col, original_end_col, [current_line])
    
    # Compatibility methods for backward compatibility
    def update_all_mappings_after_line(self, after_line: int, offset: int):
        """Deprecated: Use update_mappings_after_change instead"""
        self.update_mappings_after_change(after_line, offset)
    
    def update_line_offset(self, offset_change: int):
        """Deprecated: Offset is now handled automatically in the new methods"""
        self.cumulative_offset += offset_change
    
    def get_adjusted_line(self, original_line: int) -> int:
        """Deprecated: Use get_current_line_number instead"""
        return self.get_current_line_number(original_line)



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
    

def check_for_indentation_errors(file_path: Path) -> Tuple[bool, Optional[Tuple[str, str, int]]]:
    """Check if file has indentation errors without executing it"""
    try:
        content = file_path.read_text(encoding="utf-8")
        compile(content, str(file_path), 'exec')
        return False, None
    except IndentationError as e:
        return True, ("IndentationError", str(e), e.lineno)
    except TabError as e:
        return True, ("TabError", str(e), e.lineno)



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

def handle_indentation_error(lines: List[str], line_no: int, err_msg: str) -> List[str]:
    """Handle indentation error by adjusting the indentation of the problematic line"""
    if not line_no or line_no < 1 or line_no > len(lines):
        return lines
    
    idx = line_no - 1
    problematic_line = lines[idx]
    
    # Skip if line is empty or only whitespace
    if not problematic_line.strip():
        return lines
    
    # Determine the fix based on error message
    if "unexpected indent" in err_msg.lower():
        # Remove one level of indentation (4 spaces or 1 tab)
        if problematic_line.startswith("    "):
            lines[idx] = problematic_line[4:]
        elif problematic_line.startswith("\t"):
            lines[idx] = problematic_line[1:]
    
    elif "expected an indented block" in err_msg.lower():
        # Add one level of indentation (4 spaces)
        lines[idx] = "    " + problematic_line
    
    elif "unindent does not match" in err_msg.lower():
        # Try to align with previous non-empty line's indentation
        if idx > 0:
            # Find previous non-empty line
            prev_idx = idx - 1
            while prev_idx >= 0 and not lines[prev_idx].strip():
                prev_idx -= 1
            
            if prev_idx >= 0:
                # Get indentation of previous line
                prev_indent = len(lines[prev_idx]) - len(lines[prev_idx].lstrip())
                # Apply same indentation to current line
                lines[idx] = " " * prev_indent + problematic_line.lstrip()
    
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


latest_fix={}
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

    global latest_fix
    if target_path not in latest_fix: latest_fix[target_path]=None

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
        if debug: 
            print(f"Iteration {iteration}...")
        
        # Re-extract main lines if they were empty due to syntax errors
        if not original_main_lines:
            original_main_lines = extract_main_function_lines(target_path)
            if debug and original_main_lines:
                print(f"📊 Extracted {len(original_main_lines)} main function lines after syntax fix")
        
        # First, check for syntax errors before trying to execute
        has_syntax_error, syntax_error_info = check_for_syntax_errors(target_path)
        
        if has_syntax_error:
            iteration-=1
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

        

        has_indent_error, indent_error_info = check_for_indentation_errors(target_path)

        if has_indent_error:
            iteration-=1
            if debug: print(f"🔍 Indentation error detected!")
            err_type, err_msg, line_no = indent_error_info
            if debug: print(f"Detected {err_type} at line {line_no}: {err_msg}")
            
            # Handle indentation error (no mapping needed)
            lines = target_path.read_text(encoding="utf-8").splitlines()
            lines = handle_indentation_error(lines, line_no, err_msg)
            target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            
            if debug: print(f"✅ Applied indentation fix at line {line_no}")
            if debug: print("🔄 Retrying...\n")
            continue
        
        # If no syntax errors, try to execute the file
        if debug: print("🏃 Executing file to check for runtime errors...")
        proc = subprocess.run([sys.executable, str(target_path)], capture_output=True, text=True)

        if proc.returncode == 0:
            print(f"Fixing completed on {target_path.name}: runs without errors after {iteration} iterations")
            if debug: 
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
                sys.exit(1)
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

        try:
            if latest_fix[target_path] is not None and "lines" in fix.keys():
                if fix["lines"]==latest_fix[target_path]["lines"]: 
                    print(f"Fixing in loop: process stopped on file: {target_path}")
                    return False, mapper.mappings  # Return partial mappings even on failure
        except: pass
            
        latest_fix[target_path]=fix

        """
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
        """

        if isinstance(fix, dict):
            # Find the original column info for this line if it exists in main function
            orig_start_col = 0
            orig_end_col = 0
            
            # Look up the original column positions from original_main_lines
            if line_no in original_main_lines:
                orig_start_col, orig_end_col = original_main_lines[line_no]
            
            # Determine the number of lines in the fix
            num_fix_lines = len(fix.get('lines', [])) if 'lines' in fix else 1
            
            # Update mapping before applying the fix
            mapper.apply_fix_and_update_mapping(
                line_no, orig_start_col, orig_end_col, 
                fix['action'], num_fix_lines
            )
            
            # Apply the actual fix to the file
            lines = apply_fix_with_mapping(lines, fix, line_no, mapper, original_main_lines)
            target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            
            # Debug output
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
    print(f"Maximum iterations ({max_iterations}) reached on {target_path.name}.")
    if debug: 
        print(f"⚠️  Maximum iterations ({max_iterations}) reached.")
        print("🔧 Manual intervention may be required.")
        print(f"📊 Partial mappings created: {len(mapper.mappings)}")
    
    return False, mapper.mappings  # Return partial mappings even on failure




























































































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

smell_classes = [IdQ, IM, IQ, LC, NC, ROC]
staticDetection_smell_classes=[CG, LPQ]

thread_local = threading.local()

suppress_print=False

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
MAX_EXEC_DEPTH = 25  # You can change this value

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
        #if 'exec(' in content or 'exec ' in content:
        if 'exec(' in content:
            print(f"File {path} contains 'exec' in source code")
            return True
        
        return False
        """
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            print(f"Syntax error in {path}: {e}")
            return False  # Assume it contains exec if we can't parse it
        """
        
        # Check for exec() function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Direct exec() call
                """if isinstance(node.func, ast.Name) and node.func.id == 'exec':
                    print(f"File {path} contains exec() function call")
                    return True"""
                # builtins.exec or __builtins__.exec
                if isinstance(node.func, ast.Attribute):
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
        """if current_depth > 5:
            print(f"Skipping {file} - called from another file being analyzed")
            return []"""
        
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
                    """if current_exec_depth > max_exec_depth:
                        print(f"Detected exec at depth {current_exec_depth} (max: {max_exec_depth}) - marking for skip")
                        exec_detected.set()
                        raise RuntimeError("EXEC_DETECTED_DURING_ANALYSIS")"""
                    
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













































































import json
from pathlib import Path

def make_keys_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_keys_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_keys_serializable(i) for i in obj]
    else:
        return obj

def save_results_to_file(results: dict, file_path: Path):
    serializable_results = make_keys_serializable(results)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=4, ensure_ascii=False)
    print("Saved")









def get_function_smells(exe, executables_dict_exe):

    filename = os.path.basename(exe)  # gets "executable_initializer.py"
    if filename.startswith("executable_") and filename.endswith(".py"):
        function = filename[len("executable_"):-len(".py")]

    
    result = {}
    threads = []

    
    # Start thread for map_lines_simple
    
    """class_function_map = threading.Thread(target=wrapper, args=(map_lines_simple, result, 'map_lines_simple', executables_dict_exe, exe), kwargs={'similarity_threshold': 0.5})
    class_function_map.start()
    threads.append(class_function_map)"""
    
    

    # Start thread for map_lines_of_code
    main_map = threading.Thread(target=wrapper, args=(map_lines_of_code, result, 'map_lines_of_code', executables_dict_exe, function, exe),)
    main_map.start()
    threads.append(main_map)
    
    
    # Start thread for auto_fix_with_mapping (we keep only the mappings)
    mappings = threading.Thread(target=wrapper, args=(lambda p: auto_fix_with_mapping(p, debug=False)[1], result, 'auto_fix_with_mapping', Path(exe)))
    mappings.start()
    threads.append(mappings)
    

    # Wait for all threads to finish for this exe
    for t in threads:
        t.join()
    
    # Reduce map_lines_simple to just {generated_line: original_line}
    """
    if 'map_lines_simple' in result and 'line_mappings' in result['map_lines_simple']:
        result['map_lines_simple'] = {
            mapping['generated_line']: mapping['original_line']
            for mapping in result['map_lines_simple']['line_mappings']
            if 'generated_line' in mapping and 'original_line' in mapping
        }
    """

    # Reduce map_lines_of_code to just {generated_line: original_line}
    
    if 'map_lines_of_code' in result and 'mapping' in result['map_lines_of_code']:
        line_mappings = result['map_lines_of_code']['mapping'].get('line_mappings', [])
        result['map_lines_of_code'] = {
            mapping['generated_line']: mapping['original_line']
            for mapping in line_mappings
            if isinstance(mapping.get('generated_line'), int) and isinstance(mapping.get('original_line'), int)
        }
    
    

    # Save result for this exe
    results[exe] = result

    #save_results_to_file(results[exe], Path(f"results_dump_{function}.json"))

    
    
    auto_fix_map_var = results[exe].get("auto_fix_with_mapping", {})    # From the generated file, maps the differences of the lines before and after the fix
    #map_lines_simple_var = results[exe].get("map_lines_simple", {})     # Maps lines before the main
    map_lines_of_code_var = results[exe].get("map_lines_of_code", {})   # Maps the main lines

    print(f"Smell detection in function executable file: {exe}")
    smells = detect_smells_from_file(exe)

    print(smells)

    static_detectors_objects = [Detector(smell_cls) for smell_cls in staticDetection_smell_classes]
    for detector in static_detectors_objects:
        static_smells=detector.detect(executables_dict_exe)
        for static_smell in static_smells:
            smells.append(static_smell)



    smells = [smell for smell in smells if smell.type not in ["CG", "LPQ"]]

    for smell in smells:

        if smell.type in ["CG", "LPQ"]: pass


        if smell.type in  ["IM", "IQ", "IdQ"]:

                row = smell.row

                if row is None:
                    continue  # Skip if the smell doesn't have a row

                matched_tuple = None

                # Search for the tuple in auto_fix_with_mapping whose value list includes this row
                for tuple_str, line_list in auto_fix_map_var.items():
                    if isinstance(line_list, list) and row in line_list:
                        matched_tuple = tuple_str
                        break
                
                alternative=False

                if matched_tuple is not None:
                    #continue  # No mapping found for this smell
                    alternative=True



                if alternative==False:
                    # Convert tuple string (e.g., "(3, 0, 77)") to actual tuple
                    parsed_tuple = matched_tuple

                    if not isinstance(parsed_tuple, tuple) or len(parsed_tuple) != 3:
                        #continue
                        alternative=True

                    if alternative==False:
                        generated_line = parsed_tuple[0]

                        # Look up original line from map_lines_simple
                        original_line = map_lines_of_code_var.get(generated_line)

                        # Attach all this info to the smell object
                        smell.set_row(original_line)
                        smell.set_column_start(None)
                        smell.set_column_end(None)


                if alternative==True:
                    original_line = map_lines_of_code_var.get(row)

                    # Attach all this info to the smell object
                    smell.set_row(original_line)
                    smell.set_column_start(None)
                    smell.set_column_end(None)




        if smell.type=="LC": pass

        if smell.type == "ROC":

            rows = smell.rows

            if rows is None or not isinstance(rows, list) or len(rows) == 0:
                continue  # Skip if the smell doesn't have rows or rows is empty

            # Store the final mapped rows
            original_rows = []

            for row_tuple in rows:

                if not isinstance(row_tuple, tuple):
                    continue  # Skip if it's not a tuple
                
                # Process each row number in the tuple
                tuple_original_rows = []
                for row in row_tuple:
                    if row is None:
                        tuple_original_rows.append(None)
                        continue
                    
                    matched_tuple = None
                    
                    # Search for the tuple in auto_fix_with_mapping whose value list includes this row
                    for tuple_str, line_list in auto_fix_map_var.items():
                        if isinstance(line_list, list) and row in line_list:
                            matched_tuple = tuple_str
                            break

                    alternative=False    
                    
                    if matched_tuple is None:
                        #tuple_original_rows.append(None)  # No mapping found for this row
                        #continue
                        alternative=True
                    

                    if alternative==False:
                        # The matched_tuple should already be a tuple, not a string
                        parsed_tuple = matched_tuple
                        
                        generated_line = parsed_tuple[0]
                        
                        # Look up original line from map_lines_simple
                        original_line = map_lines_of_code_var.get(generated_line)
                        tuple_original_rows.append(original_line)

                    if alternative==True:
                        original_line = map_lines_of_code_var.get(row)
                        tuple_original_rows.append(original_line)

                
                # Add the processed tuple to our results
                original_rows.append(tuple(tuple_original_rows))


            # Only proceed if we found at least some mappings
            if not original_rows or all(all(row is None for row in row_tuple) for row_tuple in original_rows):
                continue  # No valid mappings found for any rows

            # Attach the mapped rows to the smell object 
            smell.set_rows(original_rows)
            smell.set_column_start(None)
            smell.set_column_end(None)




            

        if smell.type=="NC":
            # Pre-build a reverse lookup dictionary for faster searching
            row_to_tuple_map = {}
            
            for tuple_str, line_list in auto_fix_map_var.items():
                if isinstance(line_list, list):
                    for line in line_list:
                        row_to_tuple_map[line] = tuple_str

            def map_single_row(row):
                """Helper function to map a single row number"""
                if row is None:
                    return None
                
                # Fast lookup using pre-built dictionary
                matched_tuple = row_to_tuple_map.get(row)
                

                alternative=False
                if matched_tuple is None:
                    #return None
                    alternative=True
                
                if alternative==False:
                    # Convert tuple string to actual tuple
                    parsed_tuple = matched_tuple
                    if not isinstance(parsed_tuple, tuple) or len(parsed_tuple) != 3:
                        return None
                    
                    generated_line = parsed_tuple[0]
                    
                    # Look up original line from map_lines_simple
                    original_line = map_lines_of_code_var.get(generated_line)
                    return original_line
                
                if alternative==True:
                    original_line = map_lines_of_code_var.get(row)
                    return original_line



            # Track if we found any valid mappings
            found_mappings = False

            # Process run_calls
            run_calls = smell.run_calls if hasattr(smell, 'run_calls') else []
            if isinstance(run_calls, list):
                for call in run_calls:
                    if isinstance(call, dict) and 'row' in call:
                        original_row = map_single_row(call['row'])
                        if original_row is not None:
                            call['row'] = original_row
                            found_mappings = True

            # Process assign_parameter_calls  
            assign_parameter_calls = smell.assign_parameter_calls if hasattr(smell, 'assign_parameter_calls') else []
            if isinstance(assign_parameter_calls, list):
                for call in assign_parameter_calls:
                    if isinstance(call, dict) and 'row' in call:
                        original_row = map_single_row(call['row'])
                        if original_row is not None:
                            call['row'] = original_row
                            found_mappings = True

            # Process execute_calls (in case it becomes a list in the future)
            execute_calls = smell.execute_calls if hasattr(smell, 'execute_calls') else []
            if isinstance(execute_calls, list):
                for call in execute_calls:
                    if isinstance(call, dict) and 'row' in call:
                        original_row = map_single_row(call['row'])
                        if original_row is not None:
                            call['row'] = original_row
                            found_mappings = True

            # Process bind_parameter_calls (in case it becomes a list in the future)
            bind_parameter_calls = smell.bind_parameter_calls if hasattr(smell, 'bind_parameter_calls') else []
            if isinstance(bind_parameter_calls, list):
                for call in bind_parameter_calls:
                    if isinstance(call, dict) and 'row' in call:
                        original_row = map_single_row(call['row'])
                        if original_row is not None:
                            call['row'] = original_row
                            found_mappings = True

            # Only continue if we found at least some valid mappings
            if not found_mappings:
                continue  # No valid mappings found for any rows

            # Clear the main row attributes since they're not relevant for this smell type
            smell.set_row(None)
            smell.set_column_start(None)
            smell.set_column_end(None)


    
    return smells













# Pre-import potentially problematic modules before any threading

"""def preload_modules():
    try:
        import cirq
        #print("Pre-loaded cirq successfully")
    except ImportError:
        print("cirq not available, skipping pre-load")
    except Exception as e:
        print(f"Warning: Could not pre-load cirq: {e}")"""

# Call this before starting any threads
#preload_modules()


def clear_folder(folder_path):
    folder_path = os.path.abspath(folder_path)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Recursively delete a folder
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")




# This will store all results per exe
results = {}

def wrapper(target_fn, output_dict, key, *args, **kwargs):
    """Helper to run a function and store result in a shared dictionary."""
    output_dict[key] = target_fn(*args, **kwargs)


def autofix_map_detect( file_path:str ):

    global results

    threads = []
    executables_dict={}

    file = os.path.abspath(file_path)
    output_saving_folder=os.path.abspath("C:/Users/rical/OneDrive/Desktop/SmellResults")

    generator = FunctionExecutionGenerator()


    output_directory = "generated_executables"
    executables = generator.analyze_and_generate_all_executables(file, output_directory)

    #return 

    print(f"\nGenerated {len(executables)} executable files in '{output_directory}/' directory for {file_path} file")

    # Map each generated executable to its original file
    for exe in executables:
        # Compose full filename: executable_<function_name>.py
        exe_filename = f"executable_{exe}.py"
        
        # Join with output directory to get full path
        abs_exe_path = os.path.abspath(os.path.join(output_directory, exe_filename))
        abs_source_path = os.path.abspath(file)

        executables_dict[abs_exe_path] = abs_source_path

    smells_dict = {}
    threads = []

    # Thread-safe lock to avoid race conditions on smells_dict
    lock = threading.Lock()

    
    def process_exe(exe):
        try:
            result = get_function_smells(exe, executables_dict[exe])
            with lock:
                smells_dict[exe] = result
        except: pass

    for exe in executables_dict:
        
        try:
            print(f"Starting process in {exe}")
            thread = threading.Thread(target=process_exe, args=(exe,))
            thread.start()
            threads.append(thread)
        except Exception as e: print(f"Error while working on {exe}: {e}")

    # Wait for all threads to complete
    for t in threads:
        t.join()


    """
    print()
    print()
    print()


    for exe in smells_dict:
        print(exe)
        for smell in smells_dict[exe]:
            print(smell.as_dict())
        print()
    

    print("Qui")
    """

    file_smells=[]

    from smells.CG.CGDetector import CGDetector
    from smells.LPQ.LPQDetector import LPQDetector

    # Create instances
    cg_detector = CGDetector(CG)
    lpq_detector = LPQDetector(LPQ)

    # Call the detect method on the instances
    CG_file_smells = cg_detector.detect(file_path)
    LPQ_file_smells = lpq_detector.detect(file_path)

    for smells in smells_dict:
        for smell in smells_dict[smells]:
            file_smells.append(smell)

    # Then, add CG and LPQ smells (outside the previous loops)
    for CG_smell in CG_file_smells:
        file_smells.append(CG_smell)

    for LPQ_smell in LPQ_file_smells:
        file_smells.append(LPQ_smell)

    
    # Remove duplicates
    def make_hashable(obj):
        """Convert unhashable types to hashable ones"""
        if isinstance(obj, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(make_hashable(item) for item in obj)
        elif isinstance(obj, set):
            return tuple(sorted(make_hashable(item) for item in obj))
        else:
            return obj

    seen = set()
    unique_smells = []

    for smell in file_smells:
        # Access the dictionary attribute of the object
        smell_dict = smell.__dict__
        
        # Convert the dictionary to a hashable representation, handling nested unhashable types
        smell_tuple = make_hashable(smell_dict)
        
        if smell_tuple not in seen:
            seen.add(smell_tuple)
            unique_smells.append(smell)

    file_smells = unique_smells



    #CLEAR SECTION
    
    for ex in executables_dict:
        folder_path = os.path.dirname(ex)
        try: clear_folder(folder_path)
        except: pass
        break
    

    return file_smells
    





    """

        import pprint
        pprint.pp(executables_dict)

        
        
        for exe in executables_dict:
            try:
                auto_fix(file)
                smells = detect_smells_from_file(exe)
                if len(smells) > 0:

                    print(smells)

                    source_file = executables_dict[exe]

                    # Step 1: Get subfolder path (relative to the analyzed folder)
                    relative_source_path = os.path.relpath(source_file, folder)  # folder is the root you're analyzing
                    save_subfolder = os.path.join(output_saving_folder, relative_source_path)

                    # Step 2: Ensure the folder exists
                    os.makedirs(save_subfolder, exist_ok=True)

                    # Step 3: Clean the filename: remove "executable_" and ".py"
                    exe_basename = os.path.basename(exe)  # e.g., executable___init__.py
                    function_name = os.path.splitext(exe_basename)[0].replace("executable_", "")  # __init__

                    # Step 4: Build path for output CSV
                    output_csv_path = os.path.join(save_subfolder, f"{function_name}.csv")

                    # Step 5: Write smells to CSV
                    dict_rows = [smell.as_dict() for smell in smells]
                    if dict_rows:
                        fieldnames = sorted({k for d in dict_rows for k in d.keys()})

                        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
                            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(dict_rows)

                        print(f"Saved {len(dict_rows)} smells to {output_csv_path}")


            except Exception as e:
                print(e)





        #CLEAR SECTION

        for ex in executables_dict:
            folder_path = os.path.dirname(ex)
            clear_folder(folder_path)
            break 
    
        executables_dict.clear()
        
    """

        
        






def detect_smells_from_static_file(file):

    
    try:
        smells=autofix_map_detect(file)
        for smell in smells:
            print(smell.as_dict())
        
        return smells
    
    except: return []
    

def detect_smells_from_static_file_forJS(file):

    # Save original stdout
    original_stdout = sys.stdout
    
    try:
        # Redirect all prints to stderr for this function
        sys.stdout = sys.stderr
        
        print(f"Static detection started at {datetime.now()}")
        
        
        try:
            smells = autofix_map_detect(file)
            for smell in smells:
                print(smell.as_dict())  # This will now go to stderr
            
            # Convert smells to JSON-serializable format
            smells_dict = [smell.as_dict() for smell in smells]
            return smells_dict
            
        except Exception as e:
            print(f"Error in static detection: {str(e)}")
            return []
    
    finally:
        # Always restore original stdout
        sys.stdout = original_stdout






"""
if __name__ == "__main__":
    # python -m detection.StaticDetection.StaticMappedDetection

    smells=autofix_map_detect("C:/Users/rical/OneDrive/Desktop/QSmell_Tool/qspire/Qelm-main/QelmT.py")
    for smell in smells:
        print(smell.as_dict())
"""



"""
Aggiungere in ROC la possibilità di non avere il collegamento diretto con funzioni.
Se una funzione è richiamta, la detection deve essere fatta sulla chiamata della funzione, non sulla funzione stessa
Potenzialmente:
    - vedere se è rpesente lo smell
    - vedere se è una funzione
        - vedere la funzione stessa
        - se è presente uno smell nella funzione, puntare alle righe nella funzione
        - altrimenti puntare alla chiamata a funzione
"""