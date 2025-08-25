from smells.Explainer import Explainer
from smells.IdQ.IdQ import IdQ
from smells.utils.config_loader import get_smell_name, get_smell_description
from smells.utils.read_code import get_specific_line, get_adjacent_lines, get_operations



smell_type="IdQ"
name=get_smell_name(smell_type, "Name")
description = get_smell_description(smell_type, "Description")

introduction_prompt=f"""Suppose you are a Quantum Computing and Quantum Programmer expert (especially with python and Qiskit enviroment). You are also an expert in Code Smells, and Quantum Code Smells regarding Quantum Computing.
This is the introduction of the Quantum Code Smell {name}, also know as {smell_type}.
Its description is: {description}. \n"""

end_prompt="Explain each of these previuos steps, as if the user that is reading is now diving in Quantum Computing Realm. You can even not create a list to check these steps, you just need to touch all the topics I gave you."





@Explainer.register(IdQ)
class IdQExplainer(Explainer):
    
    def get_prompt(self, code, smell, method):

        introduction_specific_prompt="We have this smell if an operation performed on a qubit is a certain number of operations far from the previous operation on the same qubit. More context will be given in the following example."

        code_prompt=f"This is just a snippet of the code we're working on:\n {get_adjacent_lines(code, smell.row, 5, 5)}\n\n\n"

        smell_prompt=f"""Inside the code the user is writing there's a {smell_type} smell.\n"""
        smell_prompt+=f"The smell is situated on this specific line {get_specific_line(code,smell.row)}.\n"

        smell_prompt+=f"We have the smell because we have the {smell.operation_name} operation with a distance of {smell.operation_distance} operations from the previou one on the qubit {smell.qubit}."

        smell_prompt+=f"To solve the smell we could add the identity operation (id) on the qubit {smell.qubit} few operations before {smell.operation_name} (but not directly before it), even those operations involve other qubits different from {smell.qubit}. This is just one of the many ways to solve this smell."






        example_introduction_prompt="In the following code is provided an example of smelly code with this particular smell.\n"

        example_smell_promt="""
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit 

qc = QuantumCircuit (2, 2) 

qc.h(0) 

qc.h(1) # Operation distance from qubit 0 : 1
qc.h(1) # Operation distance from qubit 0 : 2
qc.h(1) # Operation distance from qubit 0 : 3
qc.h(1) # Operation distance from qubit 0 : 4

qc.h(0) # Total distance operation before re-using qubit 0 : 4

        """

        example_introduction_solution_prompt="""
This is the smelly free verion of the provided code:\n"""

        example_solution_promt="""
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit 

qc = QuantumCircuit (2, 2) 

qc.h(0) 

qc.h(1) # Operation distance from qubit 0 : 1
qc.h(1) # Operation distance from qubit 0 : 2

qc.h(0) # Total distance operation before re-using qubit 0 : 2

qc.h(1) # Operation distance from qubit 0 : 1
qc.h(1) # Operation distance from qubit 0 : 2

qc.h(0) # Total distance operation before re-using qubit 0 : 2

        """

        example_explanation=f"""
In this example (the smelly code), we can see that the user re-uses teh qubit 0 after performing other 4 operations (in this example we suppose that 3 is the maximum to consider the code not smelly).
The solution, which is smelly free, consists in using the identity matrix on the qubit 0 before doing other operations that increase the distance on qubit 0.
Of course this is just one of the many way to solve a {smell_type} smell, since it could be resolved by using other operations too that do not alter the probability of the qubit (such as measurements if the operation is terminal).
You have to secure that the operation on the qubit is actually performed not strictly before the smelly operation, since nothing would change. The fix must be done before it.
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
            method_prompt="Consider that this circuit we're working on is based off a sIdQulation of a certain code. So consider that same errors could occur during the parsing of the circuit itself."



        prompt=introduction_prompt+"\n"+introduction_specific_prompt+"\n"+method_prompt+"\n"+example_prompt+"\n"+code_prompt+"\n"+smell_prompt+"\n"+explanation_suggestion_promt
        return prompt

        
