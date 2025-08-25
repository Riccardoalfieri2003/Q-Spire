from smells.Explainer import Explainer
from smells.CG.CG import CG
from smells.utils.config_loader import get_smell_name, get_smell_description
from smells.utils.read_code import get_specific_line, get_adjacent_lines



smell_type="CG"
name=get_smell_name(smell_type, "Name")
description = get_smell_description(smell_type, "Description")

introduction_prompt=f"""Suppose you are a Quantum Computing and Quantum Programmer expert (especially with python and Qiskit enviroment). You are also an expert in Code Smells, and Quantum Code Smells regarding Quantum Computing.
This is the introduction of the Quantum Code Smell {name}, also know as {smell_type}.
Its description is: {description}. \n"""

end_prompt="Explain each of these previuos steps, as if the user that is reading is now diving in Quantum Computing Realm. You can even not create a list to check these steps, you just need to touch all the topics I gave you."





@Explainer.register(CG)
class CGExplainer(Explainer):
    
    def get_prompt(self, code, smell, method):

        introduction_specific_prompt="We have this smell if there's any occurrences of UnitaryGate,  HamiltonianGate or SingleQubitUnitary gate invoked with a matrix as input"

        code_prompt=f"This is just a snippet of the code we're working on:\n {get_adjacent_lines(code, smell.row, 10, 0)}\n\n\n"

        smell_prompt=f"""Inside the code the user is writing there's a {smell_type} smell.\n"""
        smell_prompt+=f"The smell is situated on this specific line {get_specific_line(code,smell.row)}.\n"

        """
        if smell.column_start is not None and smell.column_end is not None:
            smell_prompt+=f"To be more precise, the smell happens between in this exact part of the line: { get_specific_line(code,smell.row)[smell.column_start:smell.column_end] }\n"
        """

        smell_prompt+=f"We have the smell because we have a call on {smell.matrix}"
        if smell.qubits is not None: smell_prompt+=" on the qubits:{smell.qubits}\n"
        else: smell_prompt+="\n"






        example_introduction_prompt="In the following code is provided an example of smelly code with this particular smell.\n"

        example_smell_promt="""
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import unitary

        qc = QuantumCircuit(2)
        qc.unitary([1,0],[0,1], [0, 1]) \n
        """

        example_introduction_solution_prompt="""
This is the smelly free verion of the provided code:\n"""

        example_solution_promt="""
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import unitary

        qc = QuantumCircuit(2)
        qc.id([0,1]) \n
        """

        example_explanation=f"""
In this example, we can see that there's a call to the unitary function, passing, inside its parameters, the identity matrix and the list of affected qubits.
The solution, which is smelly free, eliminates the unitary call function and uses the id qiskit function, which manipulates the probability of the qubits just like the identity matrix.
Of course this is just one of the many way to solve a {smell_type} smell, since, as it can be read from the description of the smell, it can happen for different reasons.
Use this example just to understand how to solve the smell. Do not cite this inside the answer you'll give.\n"""

        example_prompt=example_introduction_prompt+example_smell_promt+example_introduction_solution_prompt+example_solution_promt+example_explanation
        




        explanation_suggestion_promt=f"""Knowing all this information I'd like you to:
        - Briefly explain to me what the {smell_type} smell is;
        - Explain to me what this particular smell is, regarding to the code I sent you and why it happens;
        - Give me some suggestions on how to solve the smell.
        - Give me an actual solution to remove the smell from the user's code and why this new version is correct
        """



        prompt=introduction_prompt+"\n"+introduction_specific_prompt+"\n"+example_prompt+"\n"+code_prompt+"\n"+smell_prompt+"\n"+explanation_suggestion_promt
        return prompt

        

