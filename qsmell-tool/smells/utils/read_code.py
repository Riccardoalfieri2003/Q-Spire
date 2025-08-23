def get_specific_line(code, line):
    lines = code.splitlines()
    row_number = line
    if row_number <= len(lines): specific_row = lines[row_number - 1]  # -1 because indexing starts at 0
    else: specific_row = -1
    return specific_row

def get_adjacent_lines(text: str, row_number: int, upper: int = 1, lower: int = 1, as_string: bool = True):
    """
    Extract a row and its adjacent lines from text.
    """
    lines = text.splitlines()
    n = len(lines)

    idx = row_number - 1
    if idx < 0 or idx >= n:
        raise IndexError("Row number out of range.")

    start = max(0, idx - upper)
    end = min(n, idx + lower + 1)

    result = lines[start:end]
    return "\n".join(result) if as_string else result