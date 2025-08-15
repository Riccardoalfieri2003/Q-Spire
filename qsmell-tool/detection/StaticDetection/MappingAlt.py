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


def _get_comment_lines(source_code: str) -> set:
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













#Example usage:
result = map_lines_simple(
    "C:/Users/rical/OneDrive/Desktop/QSmell_Tool/qsmell-tool/mpqp/mpqp/core/circuit.py",
    "C:/Users/rical/OneDrive/Desktop/QSmell_Tool/qsmell-tool/generated_executables/executable_initializer.py",
     similarity_threshold=0.5
)
#print_mapping_summary(result)


import pprint
for line in result:
    print(result[line]['generated_line'])
"""
# Get simple mappings
simple_maps = get_simple_line_mappings(result)
print(f"\nSimple mappings: {simple_maps}")

# Print detailed view
print_detailed_mappings(result, max_lines=1000)
"""