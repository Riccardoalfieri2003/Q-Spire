import numpy as np
from qiskit.circuit.library import *
from qiskit.quantum_info import Operator


# Creiamo un dizionario che mappa i nomi dei gate agli oggetti di Qiskit
gate_classes = {

    # 1 qubit unitari
    "x": XGate, "h": HGate, "y": YGate, "tdg": TdgGate, "id": IGate, "sx": SXGate,
    "z": ZGate, "t": TGate, "sxdg": SXdgGate, "s": SGate, "sdg": SdgGate,
    
    # 2 qubit unitari
    "ch": CHGate, "cs": CSGate, "cz": CZGate, "ecr": ECRGate, "iswap": iSwapGate,
    #"csdg": CSDGGate, 
    "dcx": DCXGate, "csx": CSXGate, "swap": SwapGate,
    "cy": CYGate, "cx": CXGate,

    # 3 qubit unitari
    "rccx": RCCXGate, "ccx": CCXGate, "ccz": CCZGate, "cswap": CSwapGate,

    # 4 qubit unitari
    #"rcccx": RCCCXGate,

    # 1 qubit non unitari
    "pauli": PauliGate, "u": UGate, "rv": RVGate, "ry": RYGate, "rx": RXGate, "rz": RZGate,
    "ms": MSGate, "r": RGate, "p": PhaseGate,

    # 2 qubit non unitari
    "crz": CRZGate, "cry": CRYGate, "cp": CPhaseGate, "crx": CRXGate, "ryy": RYYGate,
    "rzx": RZXGate, "rzz": RZZGate, "rxx": RXXGate, "mcp": MCPhaseGate, "cu": CUGate,

    # 3 qubit non unitari
    "mcx": MCXGate,
}







unitary_gates={

    # 1 qubit unitari
    "x": XGate, "h": HGate, "y": YGate, "tdg": TdgGate, "id": IGate, "sx": SXGate,
    "z": ZGate, "t": TGate, "sxdg": SXdgGate, "s": SGate, "sdg": SdgGate,
    
    # 2 qubit unitari
    "ch": CHGate, "cs": CSGate, "cz": CZGate, "ecr": ECRGate, "iswap": iSwapGate,
    #"csdg": CSDGGate, 
    "dcx": DCXGate, "csx": CSXGate, "swap": SwapGate,
    "cy": CYGate, "cx": CXGate,

    # 3 qubit unitari
    "rccx": RCCXGate, "ccx": CCXGate, "ccz": CCZGate, "cswap": CSwapGate,

    # 4 qubit unitari
    #"rcccx": RCCCXGate,
}

nonUnitary_gates={
    # 1 qubit non unitari
    "pauli": PauliGate, "u": UGate, "rv": RVGate, "ry": RYGate, "rx": RXGate, "rz": RZGate,
    "ms": MSGate, "r": RGate, "p": PhaseGate,

    # 2 qubit non unitari
    "crz": CRZGate, "cry": CRYGate, "cp": CPhaseGate, "crx": CRXGate, "ryy": RYYGate,
    "rzx": RZXGate, "rzz": RZZGate, "rxx": RXXGate, "mcp": MCPhaseGate, "cu": CUGate,

    # 3 qubit non unitari
    "mcx": MCXGate,
}









# --- UNITARY GATES ---
unitary_1q = {"x": XGate, "h": HGate, "y": YGate, "tdg": TdgGate, "id": IGate, "sx": SXGate, "z": ZGate, "t": TGate, "sxdg": SXdgGate, "s": SGate, "sdg": SdgGate}
unitary_2q = {"ch": CHGate, "cs": CSGate, "cz": CZGate, "ecr": ECRGate, "iswap": iSwapGate, "dcx": DCXGate, "csx": CSXGate, "swap": SwapGate, "cy": CYGate, "cx": CXGate,}
unitary_3q = {"rccx": RCCXGate, "ccx": CCXGate, "ccz": CCZGate, "cswap": CSwapGate,}
#unitary_4q = {"rcccx": RCCCXGate} #RCCXGate non riconosciuto da qiskit

# --- NON-UNITARY GATES ---
non_unitary_1q = {"pauli": PauliGate, "u": UGate, "rv": RVGate, "ry": RYGate, "rx": RXGate, "rz": RZGate, "ms": MSGate, "r": RGate, "p": PhaseGate,}
non_unitary_2q = {"crz": CRZGate, "cry": CRYGate, "cp": CPhaseGate, "crx": CRXGate, "ryy": RYYGate, "rzx": RZXGate, "rzz": RZZGate, "rxx": RXXGate, "mcp": MCPhaseGate, "cu": CUGate }
non_unitary_3q = {"mcx": MCXGate}
non_unitary_4q = set()  # Nessun gate non unitario a 4 qubit


# Funzione per ottenere la matrice di un gate se disponibile
def get_gate_matrix(gate):
    if gate in gate_classes: return Operator(gate_classes[gate]()).data
    return None  # Se il gate non esiste



















def compute_final_matrix(gate_sequence, gate_matrices, epsilon=1e-10):
    """Calcola il prodotto matriciale delle matrici dei gate in gate_sequence."""
    
    if gate_matrices.keys() == unitary_1q.keys(): final_matrix = np.eye(2)  # Matrice identità 2x2
    elif gate_matrices.keys() == unitary_2q.keys(): final_matrix = np.eye(4)  # Matrice identità 4x4
    elif gate_matrices.keys() == unitary_3q.keys(): final_matrix = np.eye(8)  # Matrice identità 8x8
    #elif gate_matrices.keys() == unitary_4q.keys(): final_matrix = np.eye(16)  # Matrice identità 16x16

    

    for gate in gate_sequence:
        final_matrix =gate_matrices[gate] @ final_matrix  # Moltiplicazione matriciale

    # Approssimazione degli elementi della matrice entro la tolleranza epsilon
    final_matrix = np.round(final_matrix, decimals=int(-np.log10(epsilon)))

    return final_matrix


def is_similar(matrix1, matrix2, epsilon=1e-100):
    """Controlla se due matrici sono simili entro una tolleranza epsilon, considerando i segni."""
    # Verifica se le due matrici sono simili entro la tolleranza, con segno e norma
    return np.allclose(matrix1, matrix2, atol=epsilon)

def find_most_similar_gate(result_matrix, unitary_matrices, epsilon=1e-10):
    """Trova il gate unitario più simile alla matrice risultante."""
    most_similar_gate = None
    smallest_diff = float('inf')

    for gate, unitary_matrix in unitary_matrices.items():
        if is_similar(result_matrix, unitary_matrix, epsilon):
            #print(f"La matrice risultante è simile a quella del gate {gate.upper()}.")
            return gate  # Se trova una matrice simile, restituisce il gate

    #print("La matrice risultante non corrisponde a nessun gate unitario noto.")
    return None



def is_identity_matrix(matrix):
    """Verifica se una matrice è la matrice identità."""
    n = matrix.shape[0]  # Ottieni la dimensione della matrice
    identity_matrix = np.eye(n)  # Crea la matrice identità di dimensione n x n
    return np.allclose(matrix, identity_matrix)  # Verifica se le due matrici sono uguali