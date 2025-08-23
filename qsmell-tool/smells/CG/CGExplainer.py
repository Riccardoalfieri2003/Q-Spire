from smells.Explainer import Explainer
from smells.CG.CG import CG

@Explainer.register(CG)
class CGExplainer(Explainer):
    
    def get_prompt(self, smell):
        return f"Explain why having a CG smell at row {smell.row} involving qubits "