from smells.Explainer import Explainer
from smells.LC.LC import LC
from smells.utils.config_loader import get_smell_name, get_smell_description
from smells.utils.read_code import get_specific_line, get_adjacent_lines, get_operations
from smells.utils.config_loader import get_detector_option



smell_type="LC"
name=get_smell_name(smell_type, "Name")
description = get_smell_description(smell_type, "Description")

introduction_prompt=f"""Suppose you are a Quantum Computing and Quantum Programmer expert (especially with python and Qiskit enviroment). You are also an expert in Code Smells, and Quantum Code Smells regarding Quantum Computing.
This is the introduction of the Quantum Code Smell {name}, also know as {smell_type}.
Its description is: {description}. \n"""

end_prompt="Explain each of these previuos steps, as if the user that is reading is now diving in Quantum Computing Realm. You can even not create a list to check these steps, you just need to touch all the topics I gave you."






@Explainer.register(LC)
class LCExplainer(Explainer):
    
    def get_prompt(self, code, smell, method):

        introduction_specific_prompt=f"""

The circuit analyzed for this smell is composed of these operations:
{get_operations(smell.circuit)}
        
The information needed to understand this smell are: 
- maximum number of parallel operations in the circuit, which in this specific circuit is: {smell.parallel_op};
- maximum number of operations on any single qubit in the circuit, which in this specific circuit is: {smell.lenght_op};
- maximum error in the gates of the used backed, which in this case is: {list(smell.error.values())[0]}
We have this smell if (1-error)^(parallel_op*lenght_op) is less than a certain threshold, which is {get_detector_option("LC", "threshold", fallback=0.5)}
"""

        code_prompt=""

        smell_prompt=""






        example_introduction_prompt=" "

        example_smell_promt=""" """

        example_introduction_solution_prompt=""" """

        example_solution_promt=""" """

        example_explanation=f"""
There are many way for this smell to be "mitigated". These are the most important ones:

Circuit Simplification (Algebraic / Gate Identities):
Replace sequences of gates with equivalent but shorter ones (e.g., cancel consecutive X gates, merge rotations).
Exploit commutativity to reorder operations and cancel them.
Use basis change (e.g., converting RZ + H + RZ into fewer rotations).

Decomposition into Subcircuits:
If the circuit logically contains independent parts (acting on disjoint qubit sets), run them separately instead of in one large circuit.
This only works if the algorithm doesn't require later entanglement between the subsets — but when it does, it's trickier.

Hybrid Quantum-Classical Splitting:
Push part of the computation to classical pre/post-processing.
For example, replace long quantum arithmetic blocks with classical computation where possible.

Of course, understand which approach is the best for our circuit.
In this specific smell, you don't have to completely solve it if you can't, we have just to mitigate it. Obviously if solved the better.
\n"""

        example_prompt=example_explanation
        




        explanation_suggestion_promt=f"""Knowing all this information I'd like you to:
        - Briefly explain to me what the {smell_type} smell is;
        - Explain to me what this particular smell is, regarding to the code I sent you and why it happens;
        - Give me some suggestions on how to solve the smell.
        - Give me an actual solution to mitigate the smell from the user's code and why this new version is correct
        """


        method_prompt=""
        if method=="static":
            method_prompt="Consider that this circuit we're working on is based off a simulation of a certain code. So consider that same errors could occur during the parsing of the circuit itself."



        prompt=introduction_prompt+"\n"+introduction_specific_prompt+"\n"+method_prompt+"\n"+example_prompt+"\n"+explanation_suggestion_promt+"\n"+end_prompt
        return prompt

        

