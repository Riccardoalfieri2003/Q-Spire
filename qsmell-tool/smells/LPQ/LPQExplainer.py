from smells.Explainer import Explainer
from smells.LPQ.LPQ import LPQ
from smells.utils.config_loader import get_smell_name, get_smell_description
from smells.utils.read_code import get_specific_line, get_adjacent_lines



smell_type="LPQ"
name=get_smell_name(smell_type, "Name")
description = get_smell_description(smell_type, "Description")

introduction_prompt=f"""Suppose you are a Quantum Computing and Quantum Programmer expert (especially with python and Qiskit enviroment). You are also an expert in Code Smells, and Quantum Code Smells regarding Quantum Computing.
This is the introduction of the Quantum Code Smell {name}, also know as {smell_type}.
Its description is: {description}. \n"""

end_prompt="Explain each of these previuos steps, as if the user that is reading is now diving in Quantum Computing Realm. You can even not create a list to check these steps, you just need to touch all the topics I gave you."





@Explainer.register(LPQ)
class LPQExplainer(Explainer):
    
    def get_prompt(self, code, smell, method):

        introduction_specific_prompt="We have this smell if there's any calls to the transpile function without the parameter initial layout"

        code_prompt=f"This is just a snippet of the code we're working on:\n {get_adjacent_lines(code, smell.row, 10, 0)}\n\n\n"

        smell_prompt=""






        example_introduction_prompt="In the following code is provided an example of smelly code with this particular smell.\n"

        example_smell_promt="""
        from qiskit import QuantumCircuit, transpile
        from qiskit.providers.fake_provider import FakeVigo
        qc=QuantumCircuit(3,3)
        backend = FakeVigo ()
        qc = transpile (qc, backend)

        """

        example_introduction_solution_prompt="""
This is the smelly free verion of the provided code:\n"""

        example_solution_promt="""
        from qiskit import QuantumCircuit, transpile
        from qiskit.providers.fake_provider import FakeVigo
        qc=QuantumCircuit(3,3)
        backend = FakeVigo ()
        qc = transpile (qc, backend, initial_layout=[3, 4, 2])

        """

        example_explanation=f"""
In this example, we can see that there's a call to the transpile method without the initial layout parameter.
The solution, which is smelly free, adds the initial layout parameter to the method. 
Of course the initial layout parameter is not always the same, since different backends could have different qubit disposition.
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

        

