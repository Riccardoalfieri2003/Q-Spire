from smells.Explainer import Explainer
from smells.ROC.ROC import ROC
from smells.utils.config_loader import get_smell_name, get_smell_description
from smells.utils.read_code import get_specific_line, get_adjacent_lines, get_operations



smell_type="ROC"
name=get_smell_name(smell_type, "Name")
description = get_smell_description(smell_type, "Description")

introduction_prompt=f"""Suppose you are a Quantum Computing and Quantum Programmer expert (especially with python and Qiskit enviroment). You are also an expert in Code Smells, and Quantum Code Smells regarding Quantum Computing.
This is the introduction of the Quantum Code Smell {name}, also know as {smell_type}.
Its description is: {description}. \n"""

end_prompt="Explain each of these previuos steps, as if the user that is reading is now diving in Quantum Computing Realm. You can even not create a list to check these steps, you just need to touch all the topics I gave you."





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





@Explainer.register(ROC)
class ROCExplainer(Explainer):
    
    def get_prompt(self, code, smell, method):

        smell.rows = sorted(set(num for group in smell.rows for num in group))



        introduction_specific_prompt="We have this smell if there's a number of sequential repeated sets of operations"

        code_prompt=f"This is just a snippet of the code we're working on:\n {extract_lines(code, smell.rows)}\n\n\n"

        smell_prompt=f"""Inside the code the user is writing there's a {smell_type} smell.\n"""
        smell_prompt+=f"The smell is situated on these lines: \n"
        for line in smell.rows:
            if line is not None: smell_prompt+=f"{get_specific_line(code, line)}.\n"
            else: smell_prompt+=f"The line here is not found. It should be near the previous ones.\n"

        smell_prompt+=f"""We have the smell because these operations {smell.operations} are repeated {smell.repetitions} times. 
Each operations has the name of the operation and qubits on which it is performed inside the circuit

In order to solve the smell you have to use the repeat method, which is the main method to solve the smell. 
If the smell is situated inside a for loop, identify which is the smelly for loop that creates the smell and replace it. Else look for any possible smelly leak."""

        






        example_introduction_prompt="In the following code is provided an example of smelly code with this particular smell.\n"

        example_smell_promt="""
from qiskit import QuantumCircuit

qc = QuantumCircuit (3, 3) # 3 Quantum and 3 Classical registers

hadamard = QuantumCircuit (1, name='H' )
hadamard.h(0)

measureQubit = QuantumCircuit (1, 1, name='M' )
measureQubit.measure (0, 0)

for i in range (3):
    for j in range (3):
        qc. append (hadamard, [j])
    for j in range (3) :
        qc.append (measureQubit, [j], [j])

        """

        example_introduction_solution_prompt="""
This is the smelly free verion of the provided code:\n"""

        example_solution_promt="""
from qiskit import QuantumCircuit

qc = QuantumCircuit (3, 3) # 3 Quantum and 3 Classical registers

hadamard = QuantumCircuit (1, name='H' )
hadamard.h(0)

measureQubit = QuantumCircuit (1, 1, name='M' )
measureQubit.measure (0, 0)


for j in range (3):
    qc. append (hadamard, [j])
for j in range (3) :
    qc.append (measureQubit, [j], [j])

qc.repeat(3)

        """

        example_explanation=f"""
In this example (the smelly code), we can see that the user re-uses iterates on the same operations, creating some repeated set of operations inside the circuit, making it longer.
The solution, which is smelly free, consists of the use of the operator "repeat", which prevents the repetitions inside the circuit. This was done by eliminating the for loop.
Of course this is just one of the many way to solve a {smell_type} smell, since not always this smell is found in for loops.
Use this example just to understand how to solve the smell. Do not cite this inside the answer you'll give.\n"""

        example_prompt=example_introduction_prompt+example_smell_promt+example_introduction_solution_prompt+example_solution_promt+example_explanation
        




        explanation_suggestion_promt=f"""Knowing all this information I'd like you to:
        - Briefly explain to me what the {smell_type} smell is;
        - Explain to me what this particular smell is, regarding to the code I sent you and why it happens;
        - Give me some suggestions on how to solve the smell.
        - Give me an actual solution to remove the smell from the user's code and why this new version is correct
        """

        method_prompt=""
        if method=="static":
            method_prompt="Consider that this circuit we're working on is based off a sROCulation of a certain code. So consider that same errors could occur during the parsing of the circuit itself."



        prompt=introduction_prompt+"\n"+introduction_specific_prompt+"\n"+method_prompt+"\n"+example_prompt+"\n"+code_prompt+"\n"+smell_prompt+"\n"+explanation_suggestion_promt+"\n"+end_prompt
        return prompt

        
