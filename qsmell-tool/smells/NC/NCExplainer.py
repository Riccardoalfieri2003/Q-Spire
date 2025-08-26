from smells.Explainer import Explainer
from smells.NC.NC import NC
from smells.utils.config_loader import get_smell_name, get_smell_description
from smells.utils.read_code import get_specific_line, get_adjacent_lines, get_operations

from qiskit import transpile


smell_type="NC"
name=get_smell_name(smell_type, "Name")
description = get_smell_description(smell_type, "Description")

introduction_prompt=f"""Suppose you are a Quantum Computing and Quantum Programmer expert (especially with python and Qiskit enviroment). You are also an expert in Code Smells, and Quantum Code Smells regarding Quantum Computing.
This is the introduction of the Quantum Code Smell {name}, also know as {smell_type}.
Its description is: {description}. \n"""

end_prompt="Explain each of these previuos steps, as if the user that is reading is now diving in Quantum Computing Realm. You can even not create a list to check these steps, you just need to touch all the topics I gave you."







def extract_unique_rows(smell):
    """
    Extract unique row numbers from each call type in a smell object.
    
    Args:
        smell: Dictionary or object containing call data
        
    Returns:
        tuple: (run_rows, execute_rows, assign_parameter_rows, bind_parameter_rows)
               Each element is a list of unique row numbers
    """
    
    def get_unique_rows_from_calls(calls):
        """Helper function to extract unique rows from a calls list"""
        if not calls:
            return []
        
        # Handle case where calls might be a dict (empty) or list
        if isinstance(calls, dict):
            return []
            
        # Extract row numbers and remove duplicates using set
        rows = [call.get('row') for call in calls if call.get('row') is not None]
        return sorted(list(set(rows)))  # Sort for consistent ordering
    
    # Handle both dictionary and object access
    if hasattr(smell, 'run_calls'):
        # Object access
        run_calls = smell.run_calls
        execute_calls = smell.execute_calls
        assign_parameter_calls = smell.assign_parameter_calls
        bind_parameter_calls = smell.bind_parameter_calls
    else:
        # Dictionary access
        run_calls = smell.get('run_calls', [])
        execute_calls = smell.get('execute_calls', [])
        assign_parameter_calls = smell.get('assign_parameter_calls', [])
        bind_parameter_calls = smell.get('bind_parameter_calls', [])
    
    # Extract unique rows for each call type
    run_rows = get_unique_rows_from_calls(run_calls)
    execute_rows = get_unique_rows_from_calls(execute_calls)
    assign_parameter_rows = get_unique_rows_from_calls(assign_parameter_calls)
    bind_parameter_rows = get_unique_rows_from_calls(bind_parameter_calls)
    
    return run_rows, execute_rows, assign_parameter_rows, bind_parameter_rows


def extract_lines(text, line_numbers, context=5):
    """
    Extract specific lines (by number) from a multi-line string,
    including a context of lines above and below. Inserts "..."
    where non-contiguous jumps occur.

    Parameters:
        text (str): The full string (multi-line).
        line_numbers (list[int]): Line numbers to extract (0-based).
        context (int): Number of lines above/below to include.

    Returns:
        str: Extracted lines joined with newlines, with "..." for gaps.
    """
    lines = text.splitlines()
    total = len(lines)

    # Collect all line indices to take
    indices = set()
    for num in line_numbers:
        start = max(0, num - context)
        end = min(total, num + context + 1)
        indices.update(range(start, end))

    # Sort indices
    sorted_indices = sorted(indices)

    # Build output with "..." between gaps
    extracted = []
    prev = None
    for i in sorted_indices:
        if prev is not None and i > prev + 1:
            extracted.append("...")
        extracted.append(lines[i])
        prev = i

    return "\n".join(extracted)



"""TO COMPLETE"""

@Explainer.register(NC)
class NCExplainer(Explainer):
    
    def get_prompt(self, code, smell, method):

        run_rows, execute_rows, assign_parameter_rows, bind_parameter_rows  = extract_unique_rows(smell)

        lists = [run_rows, execute_rows, assign_parameter_rows, bind_parameter_rows]
        all_rows = sorted(set().union(*lists))



        introduction_specific_prompt="We have this smell if the number of calls to run+execute methods is greater than number of calls to bind_parameters+assign_parameter methods."

        code_prompt=f"This is just a snippet of the code we're working on:\n {extract_lines(code, all_rows)}\n\n\n"

        smell_prompt=f"""Inside the code the user is writing there's a {smell_type} smell.\n"""
        smell_prompt+=f"The smell is situated on these lines: \n"
        for line in all_rows:
            if line is not None: smell_prompt+=f"{get_specific_line(code, line)}.\n"
            else: smell_prompt+=f"The line here is not found. It should be near the previous ones.\n"

        smell_prompt+=f"""We have the smell because there are: 
{smell.run_count} run calls, {smell.execute_count} run calls, which if added result in {smell.run_count+smell.execute_count} calls;
{smell.assign_count} assging parameter calls, {smell.bind_count} bind parameter calls, which if added result in {smell.assign_count+smell.bind_count} calls, which is less than the run/execute calls.

The main way to solve this smell is by using the assign_parameters function, but it alwasy not easy how to actually use it, since sometimes we could not have for loops to work with.
If the smell is situated inside a for loop, identify which is the smelly for loop that creates the smell and replace it. Else look for any possible smelly leak."""

        






        example_introduction_prompt="In the following code is provided an example of smelly code with this particular smell.\n"

        example_smell_promt="""
from qiskit import QuantumCircuit, Aer

theta_range = [0.00, 0.25, 0.50, 0.75, 1.00]
for theta_val in theta_range:
    qc = QuantumCircuit (5, 1)
    job = backend.run(qc)
        """

        example_introduction_solution_prompt="""
This is the smelly free verion of the provided code:\n"""

        example_solution_promt="""
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit import Parameter
theta = Parameter ('0' )

theta_range = [0.00, 0.25, 0.50, 0.75, 1.00]

qc= QuantumCircuit (5, 1)
circuits=[qc.bind_parameters({{theta:theta_val}}) for theta_val in theta_range]
job= backend.run(transpile(circuits,backend))

        """

        example_explanation=f"""
DO NOT CITE THIS EXAMPLE INSIDE THE ANSWER.
In this example (the smelly code), we can see that the user writes a code that has more run function execution than the bind/assign parameters functions call.
The solution, which is smelly free, consists of parameterizing the circuit in order to call less times the run function.
The main way to solve this smell is by using the assign_parameters function, but it alwasy not easy how to actually use it, since sometimes we could not have for loops to work with.
Use this example just to understand how to solve the smell.\n"""

        example_prompt=example_introduction_prompt+example_smell_promt+example_introduction_solution_prompt+example_solution_promt+example_explanation
        




        explanation_suggestion_promt=f"""Knowing all this information I'd like you to:
        - Briefly explain to me what the {smell_type} smell is;
        - Explain to me what this particular smell is, regarding to the code I sent you and why it happens;
        - Give me some suggestions on how to solve the smell.
        - Give me an actual solution to remove the smell from the user's code and why this new version is correct
        """

        method_prompt=""
        if method=="static":
            method_prompt="Consider that this circuit we're working on is based off a sNCulation of a certain code. So consider that same errors could occur during the parsing of the circuit itself."



        prompt=introduction_prompt+"\n"+introduction_specific_prompt+"\n"+method_prompt+"\n"+example_prompt+"\n"+code_prompt+"\n"+smell_prompt+"\n"+explanation_suggestion_promt
        return prompt

        
