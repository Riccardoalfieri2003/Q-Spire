from smells.Explainer import Explainer
from smells.IM.IM import IM
from smells.utils.config_loader import get_smell_name, get_smell_description
from smells.utils.read_code import get_specific_line, get_adjacent_lines, get_operations



smell_type="IM"
name=get_smell_name(smell_type, "Name")
description = get_smell_description(smell_type, "Description")

introduction_prompt=f"""Suppose you are a Quantum Computing and Quantum Programmer expert (especially with python and Qiskit enviroment). You are also an expert in Code Smells, and Quantum Code Smells regarding Quantum Computing.
This is the introduction of the Quantum Code Smell {name}, also know as {smell_type}.
Its description is: {description}. \n"""

end_prompt="Explain each of these previuos steps, as if the user that is reading is now diving in Quantum Computing Realm. You can even not create a list to check these steps, you just need to touch all the topics I gave you."





@Explainer.register(IM)
class IMExplainer(Explainer):
    
    def get_prompt(self, code, smell, method):

        introduction_specific_prompt="We have this smell if there's a non-terminal measurement on a qubit"

        code_prompt=f"This is just a snippet of the code we're working on:\n {get_adjacent_lines(code, smell.row, 5, 5)}\n\n\n"

        smell_prompt=f"""Inside the code the user is writing there's a {smell_type} smell.\n"""
        smell_prompt+=f"The smell is situated on this specific line {get_specific_line(code,smell.row)}.\n"

        smell_prompt+=f"We have the smell because we have a non terminal measurement on the qubit {smell.qubit}."






        example_introduction_prompt="In the following code is provided an example of smelly code with this particular smell.\n"

        example_smell_promt="""
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit 
qreg_q = QuantumRegister (3, 'q' ) 
creg_c = ClassicalRegister (2, 'c' ) 

qc = QuantumCircuit (qreg_q, creg_c) 

qc.h(qreg_q[0]) 
qc.measure (qreg_q[0], creg_c[0] ) 
qc.h(qreg_q[0]) 
qc.measure (qreg_q[0], creg_c[1])

        """

        example_introduction_solution_prompt="""
This is the smelly free verion of the provided code:\n"""

        example_solution_promt="""
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit 

qreg_q = QuantumRegister (3, 'q' ) 
creg_c = ClassicalRegister (2, 'c' )

qc = QuantumCircuit (qreg_q, creg_c) 

qc.h(qreg_q[0])
qc.cx(qreg_q[0], qreg_q[1]) 

qc.h(qreg_q[0]) 

qc.measure(qreg_q[0], creg_c[0]) 
qc.measure(qreg_q[1], creg_c[1])

        """

        example_explanation=f"""
In this example (the smelly code), we can see that there's an intermediate measurement on qubit 0.
The solution, which is smelly free, "copied" the value of the qubit 0 onto qubit 1 using a cx gate (it is not a complete copy, but works in this case).
This is the main way (and easy) to solve the smell. Remember! It is not enough to remove the measurement. It must be substitued with the cx gate to copy the qubit's value, as shown in the example.
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
            method_prompt="Consider that this circuit we're working on is based off a simulation of a certain code. So consider that same errors could occur during the parsing of the circuit itself."



        prompt=introduction_prompt+"\n"+introduction_specific_prompt+"\n"+method_prompt+"\n"+example_prompt+"\n"+code_prompt+"\n"+smell_prompt+"\n"+explanation_suggestion_promt+"\n"+end_prompt
        return prompt

        
