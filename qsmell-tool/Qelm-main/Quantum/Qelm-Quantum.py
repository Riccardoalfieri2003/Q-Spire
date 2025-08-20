"""
Currently Q-Quantum runs through ibm but can be utilized through any qpu

Future update coming soon
"""

import sys
import os
import json
import time
import queue
import threading
import logging
import traceback
from typing import List, Tuple, Dict
import numpy as np
import concurrent.futures
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import multiprocessing  # Added import

# Ensure Qiskit is called correctly
try:
    from qiskit import QuantumCircuit, transpile  # Added transpile import
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import QiskitRuntimeService  # Removed Sampler import
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.primitives import BackendSamplerV2  # Added import
except ImportError as e:
    messagebox.showerror("Import Error", f"Required Qiskit modules are missing: {e}")
    sys.exit(1)

# Import word_tokenize from NLTK
try:
    from nltk.tokenize import word_tokenize
    import nltk
    # Download the 'punkt' tokenizer if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
except ImportError:
    messagebox.showerror("Import Error", "Required NLTK modules are missing. Please install NLTK using 'pip install nltk'.")
    sys.exit(1)

# Optional: Import psutil for resource monitoring
try:
    import psutil
except ImportError:
    psutil = None

# Configure logging
logging.basicConfig(
    filename='qelm.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

class AdamOptimizer:
    """
    Simple Adam Optimizer implementation for parameter updates.
    """
    def __init__(self, parameters: np.ndarray, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.parameters = parameters.copy()
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = np.zeros_like(parameters)
        self.v = np.zeros_like(parameters)
        self.t = 0

    def step(self, gradients: np.ndarray):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        self.parameters -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return self.parameters

class QuantumLayerBase:
    """
    Base class for quantum layers that sets up simulators or IBM Quantum Runtime service and provides circuit building utilities.
    """
    def __init__(self, sim_method: str = 'simulator', num_threads: int = 1, enable_logging: bool = True, api_token: str = None):
        self.sim_method = sim_method.lower()
        self.num_threads = num_threads
        self.enable_logging = enable_logging
        self.api_token = api_token  # Store the API token
        if self.sim_method == 'simulator':
            self.backend = self.initialize_simulator()
            self.use_simulator = True
        elif self.sim_method == 'ibm_quantum':
            if not self.api_token:
                raise ValueError("API token must be provided for IBM Quantum backend.")
            self.service = self.initialize_service()
            self.use_simulator = False
        else:
            raise ValueError(f"Unsupported simulation method: {self.sim_method}")

    def initialize_service(self):
        """
        Initialize the QiskitRuntimeService with the provided API token.
        """
        try:
            service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=self.api_token
            )
            if self.enable_logging:
                logging.info("QuantumLayerBase: QiskitRuntimeService initialized successfully.")
            return service
        except Exception as e:
            if self.enable_logging:
                logging.error(f"QuantumLayerBase: Failed to initialize QiskitRuntimeService: {e}")
            raise e

    def initialize_backend(self):
        """
        Initialize the backend name from the service.
        """
        try:
            # List available backends
            backends = self.service.backends()
            if not backends:
                raise ValueError("No available backends found in IBM Quantum Runtime service.")

            # Specify your preferred backend or choose the first available
            preferred_backend = 'ibmq_qasm_simulator'  # Replace with your desired backend
            if preferred_backend in [backend.name for backend in backends]:
                backend_name = preferred_backend
            else:
                backend_name = backends[0].name  # Fallback to the first available backend

            if self.enable_logging:
                logging.info(f"QuantumLayerBase: Using backend '{backend_name}'.")
            return backend_name
        except Exception as e:
            if self.enable_logging:
                logging.error(f"QuantumLayerBase: Failed to initialize backend: {e}")
            raise e

    def initialize_simulator(self):
        """
        Initialize the AerSimulator.
        """
        try:
            backend = AerSimulator(method='statevector', max_parallel_threads=self.num_threads)
            if self.enable_logging:
                logging.info(f"{self.__class__.__name__}: Using AerSimulator with method='statevector'.")
            return backend
        except Exception as e:
            if self.enable_logging:
                logging.error(f"{self.__class__.__name__}: Failed to initialize AerSimulator: {e}")
            raise e

    def simulate(self, circuit: QuantumCircuit) -> np.ndarray:
        """
        Execute the quantum circuit using the chosen backend.
        """
        try:
            if self.use_simulator:
                job = self.backend.run(circuit, shots=1)
                result = job.result()
                final_state = result.get_statevector(circuit)
                return final_state.data
            else:
                # Execute on IBM Quantum hardware using QiskitRuntimeService
                if not hasattr(self, 'service') or self.service is None:
                    raise ValueError("Service is not initialized.")

                backend_name = self.initialize_backend()
                backend = self.service.backend(backend_name)

                # Initialize BackendSamplerV2 with the backend
                sampler = BackendSamplerV2(backend=backend)

                # Configure options if needed
                sampler.options.shots = 1024
                sampler.options.resilience_level = 1  # Example option

                if self.enable_logging:
                    logging.info(f"QuantumLayerBase: Running sampler on backend '{backend_name}'.")

                # **Fix**: Transpile the circuit before running to ensure supported instructions for qubit cross instructions to qpu's
                transpiled_circuit = transpile(circuit, backend=backend)

                # Run the sampler
                result = sampler.run([transpiled_circuit])
                job_result = result.result()
                counts = job_result.quasi_dists[0].get_counts()

                # Convert counts to probability distribution
                total = sum(counts.values())
                probabilities = np.array([
                    counts.get(format(i, f'0{int(np.ceil(np.log2(len(counts))) or 1)}b'), 0)
                    for i in range(len(counts))
                ])
                probabilities = probabilities / total
                return probabilities
        except Exception as e:
            if self.enable_logging:
                logging.error(f"QuantumLayerBase: Failed to execute circuit: {e}")
            raise e

class QuantumAttentionLayer(QuantumLayerBase):
    """
    Quantum Attention Layer.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, sim_method: str = 'simulator', num_threads: int = 1, enable_logging: bool = True, api_token: str = None):
        super().__init__(sim_method, num_threads, enable_logging, api_token)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        # Initialize quantum circuit or other parameters as needed

    def forward(self, x: List[int], mode: str = 'query') -> np.ndarray:
        """
        Forward pass for the attention layer.
        """
        try:
            # Example: Create a quantum circuit based on input x and mode
            circuit = QuantumCircuit(self.embed_dim, self.embed_dim)  # Added classical registers
            # Add gates based on x and mode
            # ...
            # For demonstration, apply a Hadamard gate to each qubit
            for qubit in range(self.embed_dim):
                circuit.h(qubit)
            
            # Add measurement instructions
            circuit.measure(range(self.embed_dim), range(self.embed_dim))
            
            # Simulate the circuit
            final_state = self.simulate(circuit)
            return final_state
        except Exception as e:
            if self.enable_logging:
                logging.error(f"QuantumAttentionLayer: Forward pass failed: {e}")
            raise e

    def get_all_parameters(self) -> np.ndarray:
        """
        Retrieve all parameters as a single numpy array.
        """
        # Placeholder implementation
        # Replace with actual parameter retrieval logic
        return np.array([])

    def set_all_parameters(self, params: np.ndarray):
        """
        Set all parameters from a single numpy array.
        """
        # Placeholder implementation
        # Replace with actual parameter setting logic
        pass

    def to_dict(self) -> dict:
        """
        Serialize the layer to a dictionary.
        """
        # Placeholder implementation
        return {}

    def from_dict(self, d: dict):
        """
        Deserialize the layer from a dictionary.
        """
        # Placeholder implementation
        pass

class QuantumFeedForwardLayer(QuantumLayerBase):
    """
    Quantum Feed-Forward Layer.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, sim_method: str = 'simulator', num_threads: int = 1, enable_logging: bool = True, api_token: str = None):
        super().__init__(sim_method, num_threads, enable_logging, api_token)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        # Initialize quantum circuit or other parameters as needed

    def forward(self, x: List[int], mode: str = 'w1') -> np.ndarray:
        """
        Forward pass for the feed-forward layer.
        """
        try:
            # Example: Create a quantum circuit based on input x and mode
            circuit = QuantumCircuit(self.hidden_dim, self.hidden_dim)  # Added classical registers
            # Add gates based on x and mode
            # ...
            # For demonstration, apply a Hadamard gate to each qubit
            for qubit in range(self.hidden_dim):
                circuit.h(qubit)
            
            # Add measurement instructions
            circuit.measure(range(self.hidden_dim), range(self.hidden_dim))
            
            # Simulate the circuit
            final_state = self.simulate(circuit)
            return final_state
        except Exception as e:
            if self.enable_logging:
                logging.error(f"QuantumFeedForwardLayer: Forward pass failed: {e}")
            raise e

    def get_all_parameters(self) -> np.ndarray:
        """
        Retrieve all parameters as a single numpy array.
        """
        # Placeholder implementation
        # Replace with actual parameter retrieval logic
        return np.array([])

    def set_all_parameters(self, params: np.ndarray):
        """
        Set all parameters from a single numpy array.
        """
        # Placeholder implementation
        # Replace with actual parameter setting logic
        pass

    def to_dict(self) -> dict:
        """
        Serialize the layer to a dictionary.
        """
        # Placeholder implementation
        return {}

    def from_dict(self, d: dict):
        """
        Deserialize the layer from a dictionary.
        """
        # Placeholder implementation
        pass

class QuantumLanguageModel:
    """
    Quantum-Enhanced Language Model integrating quantum attention and feed-forward layers.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, hidden_dim: int,
                 sim_method: str = 'simulator', num_threads: int = 1, enable_logging: bool = True, api_token: str = None):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.sim_method = sim_method
        self.num_threads = num_threads
        self.enable_logging = enable_logging
        self.api_token = api_token

        # Initialize embeddings and quantum layers
        self.embeddings = np.random.randn(vocab_size, embed_dim).astype(np.float32)
        self.attn = QuantumAttentionLayer(embed_dim, hidden_dim, sim_method, num_threads, enable_logging, api_token)
        self.ffn = QuantumFeedForwardLayer(embed_dim, hidden_dim, sim_method, num_threads, enable_logging, api_token)
        self.W_proj = np.random.randn(embed_dim, hidden_dim).astype(np.float32)
        self.W_out = np.random.randn(vocab_size, embed_dim).astype(np.float32)

        if self.enable_logging:
            logging.info("QuantumLanguageModel: Initialized successfully.")

    def forward(self, x: List[int], use_residual: bool = False) -> np.ndarray:
        """
        Forward pass of the language model.
        """
        try:
            embedded = self.embeddings[x]  # (num_samples, embed_dim)
            attn_output = self.attn.forward(x, mode='query')  # (embed_dim,)
            ffn_output = self.ffn.forward(x, mode='w1')  # (hidden_dim,)

            # Project and compute logits
            projected = np.dot(attn_output, self.W_proj)  # (hidden_dim,)
            if use_residual:
                projected += ffn_output
            logits = np.dot(projected, self.W_out.T)  # (vocab_size,)

            return logits
        except Exception as e:
            if self.enable_logging:
                logging.error(f"QuantumLanguageModel: Forward pass failed: {e}")
            raise e

    # Additional methods like get_all_parameters, set_all_parameters, save_model, load_model, etc.

    def get_all_parameters(self) -> np.ndarray:
        """
        Retrieve all parameters as a single numpy array.
        """
        return np.concatenate([
            self.attn.get_all_parameters(),
            self.ffn.get_all_parameters(),
            self.W_proj.flatten(),
            self.W_out.flatten()
        ])

    def set_all_parameters(self, params: np.ndarray):
        """
        Set all parameters from a single numpy array.
        """
        attn_size = self.attn.get_all_parameters().size
        ffn_size = self.ffn.get_all_parameters().size
        proj_size = self.embed_dim * self.hidden_dim
        out_size = self.vocab_size * self.embed_dim
        expected = attn_size + ffn_size + proj_size + out_size

        if params.shape[0] != expected:
            raise ValueError(f"Parameter mismatch. Expected {expected}, got {params.shape[0]}.")

        self.attn.set_all_parameters(params[:attn_size])
        self.ffn.set_all_parameters(params[attn_size:attn_size+ffn_size])
        self.W_proj = params[attn_size+ffn_size:attn_size+ffn_size+proj_size].reshape(self.embed_dim, self.hidden_dim)
        self.W_out = params[attn_size+ffn_size+proj_size:].reshape(self.vocab_size, self.embed_dim)

    def to_dict(self) -> dict:
        """
        Serialize the model to a dictionary.
        """
        return {
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "embeddings": self.embeddings.tolist(),
            "attn": self.attn.to_dict(),
            "ffn": self.ffn.to_dict(),
            "W_proj": self.W_proj.tolist(),
            "W_out": self.W_out.tolist(),
            "version": "4.0"  # Updated version
        }

    def from_dict(self, d: dict):
        """
        Deserialize the model from a dictionary.
        """
        if (d["vocab_size"] != self.vocab_size or
            d["embed_dim"] != self.embed_dim or
            d["num_heads"] != self.num_heads or
            d["hidden_dim"] != self.hidden_dim):
            raise ValueError("Model config mismatch.")

        self.embeddings = np.array(d["embeddings"], dtype=np.float32)
        self.attn.from_dict(d["attn"])
        self.ffn.from_dict(d["ffn"])
        self.W_proj = np.array(d["W_proj"], dtype=np.float32)
        self.W_out = np.array(d["W_out"], dtype=np.float32)

    def save_model(self, save_path: str):
        """
        Save the trained model and token mappings.
        """
        try:
            model_dict = self.to_dict()
            with open(save_path, 'w') as f:
                json.dump(model_dict, f)
            logging.info(f"Model saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save model: {e}")
            raise e

    def load_model(self, load_path: str):
        """
        Load a saved model and its token mappings.
        """
        try:
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"File {load_path} does not exist.")
            with open(load_path, 'r') as f:
                model_dict = json.load(f)
            if "version" not in model_dict or model_dict["version"] != "4.0":
                raise ValueError("Unsupported model version.")
            self.from_dict(model_dict)
            logging.info(f"Model loaded from {load_path}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise e

    def shift_parameter(self, param_index: int, shift: float):
        """
        Shift a specific parameter by a given amount.
        """
        shifted_params = self.get_all_parameters()
        shifted_params[param_index] += shift
        self.set_all_parameters(shifted_params)

    def unshift_parameter(self, param_index: int, shift: float):
        """
        Restore a specific parameter by unshifting it.
        """
        self.shift_parameter(param_index, -shift)

def create_synthetic_dataset(vocab_size: int, num_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic dataset for training.
    """
    X = np.random.randint(4, vocab_size, size=(num_samples,))  # Start from index 4 to reserve special tokens
    Y = np.random.randint(4, vocab_size, size=(num_samples,))
    return X, Y

def load_real_dataset(file_path: str, vocab_size: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Load and preprocess a real dataset from a text file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = word_tokenize(text.lower())
    freq = defaultdict(int)
    for token in tokens:
        freq[token] += 1

    # Add special tokens
    special_tokens = ["<PAD>", "<START>", "<END>", "<UNK>"]
    sorted_tokens = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    # Reserve indices for special tokens
    token_to_id = {token: idx for idx, token in enumerate(special_tokens)}
    for token, _ in sorted_tokens:
        if len(token_to_id) >= vocab_size:
            break
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    X, Y_ids = [], []
    for i in range(len(tokens)-1):
        current_token = tokens[i]
        next_token = tokens[i+1]
        X.append(token_to_id.get(current_token, token_to_id["<UNK>"]))
        Y_ids.append(token_to_id.get(next_token, token_to_id["<UNK>"]))

    Y = np.array(Y_ids, dtype=np.int32)

    return np.array(X), Y, token_to_id

def cross_entropy_loss(logits: np.ndarray, target: int) -> float:
    """
    Cross-Entropy Loss for next-token prediction.
    """
    # Numerical stability
    logits = logits - np.max(logits)
    softmax = np.exp(logits) / np.sum(np.exp(logits))
    # Avoid log(0)
    softmax = np.clip(softmax, 1e-12, 1.0)
    return -np.log(softmax[target])

def perplexity(logits: np.ndarray, target: int) -> float:
    """
    Calculate Perplexity for a single prediction.
    """
    ce_loss = cross_entropy_loss(logits, target)
    return np.exp(ce_loss)

def bleu_score(reference: List[str], hypothesis: List[str], max_n: int = 4) -> float:
    """
    Compute BLEU score between reference and hypothesis.
    Simplistic implementation for demonstration purposes.
    """
    from collections import Counter
    import math

    weights = [1.0 / max_n] * max_n
    reference_counts = [Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)]) for n in range(1, max_n+1)]
    hypothesis_counts = [Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)]) for n in range(1, max_n+1)]

    precisions = []
    for ref_count, hyp_count in zip(reference_counts, hypothesis_counts):
        overlap = hyp_count & ref_count
        precision = sum(overlap.values()) / max(sum(hyp_count.values()), 1e-12)
        precisions.append(precision)

    # Brevity penalty
    ref_length = len(reference)
    hyp_length = len(hypothesis)
    if hyp_length == 0:
        bp = 0
    elif hyp_length > ref_length:
        bp = 1
    else:
        bp = math.exp(1 - ref_length / hyp_length)

    # Geometric mean of precisions
    if min(precisions) > 0:
        log_precisions = [w * math.log(p) for w, p in zip(weights, precisions)]
        geo_mean = math.exp(sum(log_precisions))
    else:
        geo_mean = 0

    bleu = bp * geo_mean
    return bleu

def compute_gradient_for_parameter(args):
    """
    Compute gradient for a single parameter using the parameter-shift rule with Cross-Entropy Loss.
    Intended for parallel execution.
    """
    (vocab_size, embed_dim, num_heads, hidden_dim, sim_method, num_threads, X, Y, original_params, i, api_token) = args
    try:
        model = QuantumLanguageModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            sim_method=sim_method,
            num_threads=num_threads,
            enable_logging=False,  # Suppress logging during gradient computation
            api_token=api_token
        )
        model.set_all_parameters(original_params)

        # Parameter shift
        shift = np.pi / 2
        model.shift_parameter(i, shift)
        loss_plus = np.mean([
            cross_entropy_loss(model.forward([x], use_residual=True), y)
            for x, y in zip(X, Y)
        ])

        model.unshift_parameter(i, shift)
        model.shift_parameter(i, -shift)
        loss_minus = np.mean([
            cross_entropy_loss(model.forward([x], use_residual=True), y)
            for x, y in zip(X, Y)
        ])

        model.unshift_parameter(i, -shift)

        gradient = (loss_plus - loss_minus) / 2.0
        return i, gradient
    except Exception:
        traceback.print_exc()
        return i, 0.0

def compute_gradients_parallel(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray, num_processes: int = 1, progress_callback=None) -> np.ndarray:
    """
    Compute gradients for all parameters in parallel.
    Calls progress_callback(completed, total, param_index, gradient) for each computed gradient.
    """
    gradients = np.zeros_like(model.get_all_parameters())
    original_params = model.get_all_parameters().copy()
    total_params = len(original_params)

    # Define block size for logging
    block_size = 10000  # Adjust as needed
    blocks = (total_params + block_size - 1) // block_size

    args_list = [
        (
            model.vocab_size,
            model.embed_dim,
            model.num_heads,
            model.hidden_dim,
            model.sim_method,
            model.num_threads,
            X,
            Y,
            original_params,
            i,
            model.api_token
        )
        for i in range(total_params)
    ]

    # Parallel execution with progress callback
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(compute_gradient_for_parameter, args): args[-2] for args in args_list}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            i, gradient = future.result()
            gradients[i] = gradient
            completed += 1
            # Log progress every block_size parameters
            if completed % block_size == 0 or completed == total_params:
                if progress_callback:
                    progress_callback(completed, total_params, i, gradient)

    return gradients

def train_model(model: QuantumLanguageModel, X: np.ndarray, Y: np.ndarray,
                epochs: int = 10, lr: float = 0.001, num_threads: int = 1,
                log_queue: queue.Queue = None, stop_flag=None, time_lock: threading.Lock = None, time_data=None,
                optimizer=None):
    """
    Train the Quantum Language Model using the provided optimizer.
    """
    if time_data is None:
        time_data = {}
    start_time = time_data['start_time'] = time.time()
    time_data['epochs_done'] = 0
    time_data['epochs'] = epochs

    for epoch in range(epochs):
        if stop_flag and stop_flag.is_set():
            if log_queue:
                log_queue.put("Training stopped by user.\n")
            break

        if log_queue:
            log_queue.put(f"Starting Epoch {epoch+1}/{epochs}\n")

        epoch_start_time = time.time()

        def progress_callback(completed, total, param_index, gradient):
            """
            Callback function to update progress.
            """
            progress = (completed / total) * 100
            if log_queue:
                log_queue.put(f"Gradient Computation Progress: {completed}/{total} parameters ({progress:.2f}%) completed.\n")
            # Since we are in a separate thread, use thread-safe method to update GUI
            # This requires passing a reference or using thread-safe queues
            # For simplicity, we can send the progress value through the log_queue and parse it in the GUI
            # Alternatively, use another queue or shared variable
            # Here, we'll skip updating the progress bar via this callback

        gradients = compute_gradients_parallel(model, X, Y, num_processes=num_threads, progress_callback=progress_callback)

        # Update parameters using optimizer
        if optimizer:
            updated_params = optimizer.step(gradients)
            model.set_all_parameters(updated_params)
        else:
            # Fallback to simple gradient descent
            params = model.get_all_parameters()
            params -= lr * gradients
            model.set_all_parameters(params)

        # Compute average loss and perplexity
        total_loss = np.mean([
            cross_entropy_loss(model.forward([x], use_residual=True), y)
            for x, y in zip(X, Y)
        ])
        total_perplexity = np.mean([
            perplexity(model.forward([x], use_residual=True), y)
            for x, y in zip(X, Y)
        ])

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        # Update GUI log
        if log_queue:
            log_queue.put(f"Epoch {epoch+1}/{epochs} completed in {epoch_duration:.2f}s, Average Loss: {total_loss:.6f}, Perplexity: {total_perplexity:.6f}\n")

        # Update evaluation metrics (BLEU score can be computed on a validation set if available)

        if time_lock:
            with time_lock:
                time_data['epochs_done'] = epoch + 1
                elapsed = epoch_end_time - start_time
                if time_data['epochs_done'] > 0 and time_data['epochs_done'] < epochs:
                    per_epoch = elapsed / time_data['epochs_done']
                    remaining = (epochs - time_data['epochs_done']) * per_epoch
                    time_data['remaining'] = remaining
                else:
                    time_data['remaining'] = 0

        # Update the epoch_progress bar
        epoch_progress = ((epoch + 1) / epochs) * 100
        if epoch_progress > 100:
            epoch_progress = 100
        if log_queue:
            log_queue.put(f"Epoch Progress: {epoch_progress:.2f}%\n")

    if log_queue and (not stop_flag or not stop_flag.is_set()):
        log_queue.put("Training completed.\n")

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax values for each set of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_inference(model: QuantumLanguageModel, input_sequence: List[int], token_to_id: Dict[str, int], id_to_token: Dict[int, str], max_length: int = 50, temperature: float = 1.0, log_callback=None):
    """
    Generate a sequence of tokens based on the input sequence.
    """
    try:
        generated = input_sequence.copy()
        for _ in range(max_length):
            logits = model.forward([generated[-1]], use_residual=True)
            probabilities = softmax(logits / temperature)

            # Sample from the probability distribution
            chosen_index = np.random.choice(len(probabilities), p=probabilities)
            generated.append(chosen_index)

            # Stop generation if <END> token is generated
            if chosen_index == token_to_id.get("<END>", chosen_index):
                break

        # Convert token IDs back to tokens
        generated_tokens = [id_to_token.get(idx, "<UNK>") for idx in generated]
        response = " ".join(generated_tokens)

        if log_callback:
            log_callback(f"Generated Response:\n{response}\n\n")

        return generated_tokens, response
    except Exception as e:
        if log_callback:
            log_callback(f"Inference failed: {e}\n")
        raise e

class QELM_GUI:
    """
    Graphical User Interface for the Quantum-Enhanced Language Model.
    """
    def __init__(self, master):
        try:
            self.master = master
            master.title("QELM Trainer - Enhanced")
            master.geometry("1600x1000")  # Increased window size for better layout *edit based on your screen size*
            master.resizable(False, False)

            # Default Model parameters
            self.vocab_size = 10000  # Default vocabulary size
            self.embed_dim = 256      # Default embedding dimension
            self.num_heads = 8        # Default number of attention heads
            self.hidden_dim = 512     # Default hidden dimension
            self.sim_method = 'simulator'
            self.num_threads = min(8, multiprocessing.cpu_count())  # Adjusted for higher model complexity
            self.api_token = ""  # Initialize API token as empty string (Don't store your API here, pop it in the box and let it remain hidden).

            self.model = QuantumLanguageModel(
                self.vocab_size, 
                self.embed_dim, 
                self.num_heads, 
                self.hidden_dim,
                sim_method=self.sim_method, 
                num_threads=self.num_threads, 
                enable_logging=True,
                api_token=self.api_token  # Pass the API token
            )

            self.token_to_id = {}
            self.id_to_token = {}

            # Initialize optimizer
            self.optimizer = AdamOptimizer(self.model.get_all_parameters(), lr=0.001)

            # Training controls
            self.stop_flag = threading.Event()
            self.time_data = {'start_time': 0, 'epochs_done': 0, 'remaining': 0, 'epochs': 0}
            self.time_lock = threading.Lock()

            # Initialize per-process CPU usage monitoring
            if psutil:
                self.process = psutil.Process(os.getpid())
                self.process.cpu_percent(interval=None)  # Initialize
            else:
                self.process = None

            # Configure GUI appearance to whatever suits your mood (Please keep gradio away from my code).
            self.master.configure(bg="#2C3E50")
            style = ttk.Style(self.master)
            style.theme_use('clam')
            style.configure(".", background="#2C3E50", foreground="white")
            style.configure("TFrame", background="#2C3E50")
            style.configure("TLabelFrame", background="#34495E", foreground="white")
            style.configure("TLabel", background="#2C3E50", foreground="white")
            style.configure("TButton", background="#34495E", foreground="white", padding=6, relief="flat")
            style.configure("TNotebook", background="#2C3E50")
            style.configure("TNotebook.Tab", background="#34495E", foreground="white")
            style.configure("Horizontal.TProgressbar", background="#1ABC9C", troughcolor="#34495E")
            # Lighter background for Entry/Spinbox
            style.configure("Custom.TEntry", fieldbackground="#455A64", foreground="white", insertcolor="white")
            style.configure("TSpinbox", fieldbackground="#455A64", foreground="white")
            style.map("TButton", foreground=[('active', 'white')], background=[('active', '#1F2A36')])

            self.create_widgets()
            self.update_resource_usage()
            self.update_time_label()

            # Initialize log queue and start log handler
            self.log_queue = queue.Queue()
            self.master.after(100, self.process_log_queue)

        except Exception as e:
            error_trace = traceback.format_exc()
            logging.critical(f"GUI Initialization error:\n{error_trace}")
            messagebox.showerror("Initialization Error", f"An error occurred during GUI initialization:\n{e}\n\nCheck the log file for more details.")
            sys.exit(1)

    def create_widgets(self):
        """
        Create and arrange all widgets in the GUI.
        """
        container = ttk.Frame(self.master)
        container.pack(fill='both', expand=True)

        left_frame = ttk.Frame(container)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        right_frame = ttk.Frame(container)
        right_frame.pack(side='right', fill='y', padx=10, pady=10)

        self.notebook = ttk.Notebook(left_frame)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_infer = ttk.Frame(self.notebook)
        self.tab_manage = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_train, text='Train Model')
        self.notebook.add(self.tab_infer, text='Run Inference')
        self.notebook.add(self.tab_manage, text='Manage Token Mappings')
        self.notebook.pack(fill='both', expand=True)

        # =======================
        # Train Model Tab
        # =======================
        dataset_frame = ttk.LabelFrame(self.tab_train, text="Dataset Selection")
        dataset_frame.pack(fill='x', padx=10, pady=10)

        self.dataset_path_var = tk.StringVar(value="No dataset selected.")
        ttk.Label(dataset_frame, textvariable=self.dataset_path_var).pack(side='left', padx=10, pady=10)
        select_dataset_btn = ttk.Button(dataset_frame, text="Select Dataset", command=self.select_dataset)
        select_dataset_btn.pack(side='right', padx=10, pady=10)

        hyperparams_frame = ttk.LabelFrame(self.tab_train, text="Model Parameters")
        hyperparams_frame.pack(fill='x', padx=10, pady=10)

        # Vocabulary Size
        ttk.Label(hyperparams_frame, text="Vocabulary Size:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.vocab_size_entry = ttk.Entry(hyperparams_frame, width=15, style="Custom.TEntry")
        self.vocab_size_entry.insert(0, str(self.vocab_size))
        self.vocab_size_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        # Embedding Dimension
        ttk.Label(hyperparams_frame, text="Embedding Dimension:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.embed_dim_entry = ttk.Entry(hyperparams_frame, width=15, style="Custom.TEntry")
        self.embed_dim_entry.insert(0, str(self.embed_dim))
        self.embed_dim_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        # Number of Attention Heads
        ttk.Label(hyperparams_frame, text="Number of Attention Heads:").grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.num_heads_entry = ttk.Entry(hyperparams_frame, width=15, style="Custom.TEntry")
        self.num_heads_entry.insert(0, str(self.num_heads))
        self.num_heads_entry.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        # Hidden Dimension
        ttk.Label(hyperparams_frame, text="Hidden Dimension:").grid(row=3, column=0, padx=10, pady=10, sticky='e')
        self.hidden_dim_entry = ttk.Entry(hyperparams_frame, width=15, style="Custom.TEntry")
        self.hidden_dim_entry.insert(0, str(self.hidden_dim))
        self.hidden_dim_entry.grid(row=3, column=1, padx=10, pady=10, sticky='w')

        # Learning Rate and Epochs
        ttk.Label(hyperparams_frame, text="Learning Rate:").grid(row=4, column=0, padx=10, pady=10, sticky='e')
        self.lr_entry = ttk.Entry(hyperparams_frame, width=15, style="Custom.TEntry")
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.grid(row=4, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(hyperparams_frame, text="Epochs:").grid(row=5, column=0, padx=10, pady=10, sticky='e')
        self.epochs_entry = ttk.Entry(hyperparams_frame, width=15, style="Custom.TEntry")
        self.epochs_entry.insert(0, "1")  # Reduced epochs for testing
        self.epochs_entry.grid(row=5, column=1, padx=10, pady=10, sticky='w')

        sim_settings_frame = ttk.LabelFrame(self.tab_train, text="Execution Settings")
        sim_settings_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(sim_settings_frame, text="Execution Backend:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.backend_var = tk.StringVar(value="simulator")  # Options: 'simulator', 'ibm_quantum'
        simulator_radio = ttk.Radiobutton(sim_settings_frame, text='Simulator', variable=self.backend_var, value='simulator', command=self.update_backend_selection)
        ibm_radio = ttk.Radiobutton(sim_settings_frame, text='IBM Quantum', variable=self.backend_var, value='ibm_quantum', command=self.update_backend_selection)
        simulator_radio.grid(row=0, column=1, padx=10, pady=10, sticky='w')
        ibm_radio.grid(row=0, column=2, padx=10, pady=10, sticky='w')

        # API Token Entry (Initially hidden)
        self.api_token_label = ttk.Label(sim_settings_frame, text="IBM Quantum API Token:")
        self.api_token_entry = ttk.Entry(sim_settings_frame, width=50, style="Custom.TEntry", show="*")  # Show as password
        self.api_token_label.grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.api_token_entry.grid(row=1, column=1, columnspan=2, padx=10, pady=10, sticky='w')
        self.api_token_label.grid_remove()
        self.api_token_entry.grid_remove()

        ttk.Label(sim_settings_frame, text="Number of Threads:").grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.num_threads_var = tk.IntVar(value=self.num_threads)
        self.num_threads_spinbox = ttk.Spinbox(
            sim_settings_frame,
            from_=1,
            to=multiprocessing.cpu_count(),
            textvariable=self.num_threads_var,
            width=5
        )
        self.num_threads_spinbox.grid(row=2, column=1, padx=10, pady=10, sticky='w')
        ttk.Label(sim_settings_frame, text=f"(Max: {multiprocessing.cpu_count()})").grid(row=2, column=2, padx=10, pady=10, sticky='w')

        train_controls_frame = ttk.Frame(self.tab_train)
        train_controls_frame.pack(fill='x', padx=10, pady=10)

        self.train_button = ttk.Button(train_controls_frame, text="Start Training", command=self.train_model)
        self.train_button.pack(side='left', padx=10, pady=10)

        stop_button = ttk.Button(train_controls_frame, text="STOP (Graceful)", command=self.stop_training)
        stop_button.pack(side='left', padx=10, pady=10)

        hard_stop_button = ttk.Button(train_controls_frame, text="HARD STOP (Immediate)", command=self.hard_stop)
        hard_stop_button.pack(side='left', padx=10, pady=10)

        self.save_button = ttk.Button(train_controls_frame, text="Save Model", command=self.save_model)
        self.save_button.pack(side='left', padx=10, pady=10)

        self.load_button = ttk.Button(train_controls_frame, text="Load Model", command=self.load_model)
        self.load_button.pack(side='left', padx=10, pady=10)

        # Progress Bars
        progress_bars_frame = ttk.Frame(self.tab_train)
        progress_bars_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(progress_bars_frame, text="Training Progress:").pack(anchor='w', padx=10, pady=5)
        self.epoch_progress = ttk.Progressbar(progress_bars_frame, orient='horizontal', length=600, mode='determinate')
        self.epoch_progress.pack(fill='x', padx=10, pady=5)

        ttk.Label(progress_bars_frame, text="Gradient Computation Progress:").pack(anchor='w', padx=10, pady=5)
        self.gradient_progress = ttk.Progressbar(progress_bars_frame, orient='horizontal', length=600, mode='determinate')
        self.gradient_progress.pack(fill='x', padx=10, pady=5)

        log_frame = ttk.LabelFrame(self.tab_train, text="Training Log")
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.train_log = scrolledtext.ScrolledText(log_frame, state='disabled', wrap='word', font=("Courier", 10),
                                                 bg="#2C3E50", fg="white", insertbackground="white")
        self.train_log.pack(fill='both', expand=True, padx=5, pady=5)

        # Evaluation Metrics within Train Model Tab
        eval_metrics_frame = ttk.LabelFrame(self.tab_train, text="Evaluation Metrics")
        eval_metrics_frame.pack(fill='x', padx=10, pady=10)

        self.perplexity_label = ttk.Label(eval_metrics_frame, text="Perplexity: N/A")
        self.perplexity_label.pack(anchor='w', padx=10, pady=5)

        self.bleu_label = ttk.Label(eval_metrics_frame, text="BLEU Score: N/A")
        self.bleu_label.pack(anchor='w', padx=10, pady=5)

        # =======================
        # Run Inference Tab
        # =======================
        inference_frame = ttk.LabelFrame(self.tab_infer, text="Inference")
        inference_frame.pack(fill='x', padx=10, pady=10)

        ttk.Label(inference_frame, text="Input Token:").grid(row=0, column=0, padx=10, pady=10, sticky='e')
        self.input_token_entry = ttk.Entry(inference_frame, width=30, style="Custom.TEntry")
        self.input_token_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(inference_frame, text="Max Length:").grid(row=1, column=0, padx=10, pady=10, sticky='e')
        self.max_length_entry = ttk.Entry(inference_frame, width=15, style="Custom.TEntry")
        self.max_length_entry.insert(0, "50")
        self.max_length_entry.grid(row=1, column=1, padx=10, pady=10, sticky='w')

        ttk.Label(inference_frame, text="Temperature:").grid(row=2, column=0, padx=10, pady=10, sticky='e')
        self.temperature_entry = ttk.Entry(inference_frame, width=15, style="Custom.TEntry")
        self.temperature_entry.insert(0, "1.0")
        self.temperature_entry.grid(row=2, column=1, padx=10, pady=10, sticky='w')

        inference_controls_frame = ttk.Frame(self.tab_infer)
        inference_controls_frame.pack(fill='x', padx=10, pady=10)

        self.infer_button = ttk.Button(inference_controls_frame, text="Run Inference", command=self.run_inference)
        self.infer_button.pack(side='left', padx=10, pady=10)

        infer_log_frame = ttk.LabelFrame(self.tab_infer, text="Inference Output")
        infer_log_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.infer_log = scrolledtext.ScrolledText(infer_log_frame, state='disabled', wrap='word', font=("Courier", 10),
                                                 bg="#2C3E50", fg="white", insertbackground="white")
        self.infer_log.pack(fill='both', expand=True, padx=5, pady=5)

        # =======================
        # Manage Token Mappings Tab
        # =======================
        token_map_frame = ttk.LabelFrame(self.tab_manage, text="Token Mappings")
        token_map_frame.pack(fill='both', expand=True, padx=10, pady=10)

        load_token_map_button = ttk.Button(token_map_frame, text="Load Token Map", command=self.load_token_map)
        load_token_map_button.pack(side='top', padx=10, pady=10)

        self.token_map_display = scrolledtext.ScrolledText(token_map_frame, state='disabled', wrap='word', font=("Courier", 10),
                                                          bg="#2C3E50", fg="white", insertbackground="white")
        self.token_map_display.pack(fill='both', expand=True, padx=5, pady=5)

        # =======================
        # System Resources & Time
        # =======================
        usage_frame = ttk.LabelFrame(right_frame, text="System Resources & Time")
        usage_frame.pack(fill='y', padx=5, pady=5)

        self.cpu_label = ttk.Label(usage_frame, text="CPU: N/A")
        self.cpu_label.pack(anchor='w', padx=10, pady=5)

        self.gpu_label = ttk.Label(usage_frame, text="GPU: Check externally (e.g., nvidia-smi)")
        self.gpu_label.pack(anchor='w', padx=10, pady=5)

        self.time_label = ttk.Label(usage_frame, text="Elapsed: 0s | Remaining: Estimating...")
        self.time_label.pack(anchor='w', padx=10, pady=5)

    def process_log_queue(self):
        """
        Process messages from the log queue and update the GUI accordingly.
        """
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                # Append all messages to the training log
                self.train_log.config(state='normal')
                self.train_log.insert(tk.END, message)
                self.train_log.see(tk.END)
                self.train_log.config(state='disabled')
        except Exception as e:
            logging.error(f"Error processing log queue: {e}")
        finally:
            self.master.after(100, self.process_log_queue)  # Continue polling

    def update_backend_selection(self):
        """
        Update the GUI or internal settings based on backend selection.
        Show or hide the API token entry field.
        """
        selected_backend = self.backend_var.get()
        if selected_backend == 'ibm_quantum':
            self.api_token_label.grid()
            self.api_token_entry.grid()
            self.log_train("Selected Execution Backend: IBM Quantum Hardware.\n")
        else:
            self.api_token_label.grid_remove()
            self.api_token_entry.grid_remove()
            self.log_train("Selected Execution Backend: Simulator.\n")

    def log_train(self, message: str):
        """
        Log messages to the training log via the queue.
        """
        if hasattr(self, 'log_queue'):
            self.log_queue.put(message)

    def log_infer(self, message: str):
        """
        Log messages to the inference log.
        """
        self.infer_log.config(state='normal')
        self.infer_log.insert(tk.END, message)
        self.infer_log.see(tk.END)
        self.infer_log.config(state='disabled')

    def log_token_map(self, message: str):
        """
        Log messages to the token map display.
        """
        self.token_map_display.config(state='normal')
        self.token_map_display.insert(tk.END, message)
        self.token_map_display.see(tk.END)
        self.token_map_display.config(state='disabled')

    def select_dataset(self):
        """
        Open a file dialog to select a dataset.
        """
        try:
            file_path = filedialog.askopenfilename(title="Select Dataset File",
                                                   filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
            if file_path:
                self.dataset_path = file_path
                self.dataset_path_var.set(file_path)
                self.log_train(f"Selected Dataset: {file_path}\n")
                self.token_to_id = {}
        except Exception as e:
            err_msg = f"Error selecting dataset:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Dataset Selection Error", err_msg)

    def train_model(self):
        """
        Start the training process in a separate thread.
        """
        try:
            # Retrieve and validate model parameters from GUI
            vocab_size = int(self.vocab_size_entry.get())
            embed_dim = int(self.embed_dim_entry.get())
            num_heads = int(self.num_heads_entry.get())
            hidden_dim = int(self.hidden_dim_entry.get())
            lr = float(self.lr_entry.get())
            epochs = int(self.epochs_entry.get())

            if vocab_size <= 0 or embed_dim <= 0 or num_heads <= 0 or hidden_dim <= 0 or lr <= 0 or epochs <= 0:
                raise ValueError

            if embed_dim % num_heads != 0:
                messagebox.showerror("Invalid Input", "Embedding Dimension must be divisible by Number of Attention Heads.")
                return

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid positive integers for model parameters and epochs, and a positive float for learning rate.")
            return

        sim_method = self.backend_var.get()
        num_threads = self.num_threads_var.get()
        max_threads = multiprocessing.cpu_count()
        if num_threads > max_threads:
            messagebox.showwarning("Thread Limit", f"Resetting threads to max {max_threads}")
            num_threads = max_threads
            self.num_threads_var.set(num_threads)

        # If IBM Quantum backend is selected, retrieve the API token
        if sim_method == 'ibm_quantum':
            api_token = self.api_token_entry.get().strip()
            if not api_token:
                messagebox.showerror("API Token Missing", "Please enter your IBM Quantum API Token.")
                return
            self.api_token = api_token
        else:
            self.api_token = ""  # Reset API token if not using IBM Quantum

        # Load dataset
        if hasattr(self, 'dataset_path') and self.dataset_path:
            dataset_path = self.dataset_path
            try:
                X, Y, token_to_id = load_real_dataset(dataset_path, vocab_size)
                self.X = X
                self.Y = Y
                self.token_to_id = token_to_id
                self.id_to_token = {idx: token for token, idx in token_to_id.items()}
                self.log_train(f"Loaded real dataset from {dataset_path}\n")
            except Exception as e:
                err_msg = f"Failed to load dataset:\n{traceback.format_exc()}"
                self.log_train(err_msg + "\n")
                messagebox.showerror("Dataset Load Error", err_msg)
                return
        else:
            X, Y = create_synthetic_dataset(vocab_size, num_samples=500)  # Can be changed but utilizing a real dataset is best
            self.X = X
            self.Y = Y
            self.log_train("Using synthetic dataset for training.\n")

        # Reinitialize the model with new parameters
        try:
            self.model = QuantumLanguageModel(
                vocab_size, 
                embed_dim, 
                num_heads, 
                hidden_dim,
                sim_method=sim_method, 
                num_threads=num_threads, 
                enable_logging=True,
                api_token=self.api_token
            )
            self.optimizer = AdamOptimizer(self.model.get_all_parameters(), lr=lr)
            self.log_train("Model re-initialized with new parameters.\n")
        except Exception as e:
            err_msg = f"Failed to initialize model with new parameters:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Model Initialization Error", err_msg)
            return

        # Update model execution settings
        self.model.sim_method = sim_method
        self.model.num_threads = num_threads
        self.model.attn.sim_method = sim_method
        self.model.attn.num_threads = num_threads
        if sim_method == 'ibm_quantum':
            # Ensure that the backend is initialized for IBM Quantum
            try:
                self.model.attn.service = self.model.attn.initialize_service()
                self.model.attn.use_simulator = False
                self.model.ffn.service = self.model.ffn.initialize_service()
                self.model.ffn.use_simulator = False
            except Exception as e:
                err_msg = f"Failed to initialize IBM Quantum service:\n{traceback.format_exc()}"
                self.log_train(err_msg + "\n")
                messagebox.showerror("IBM Quantum Initialization Error", err_msg)
                return
        else:
            self.model.attn.backend = self.model.attn.initialize_simulator()
            self.model.ffn.backend = self.model.ffn.initialize_simulator()

        # Disable buttons during training
        self.train_button.config(state='disabled')
        self.save_button.config(state='disabled')
        self.load_button.config(state='disabled')
        self.infer_button.config(state='disabled')
        self.stop_flag.clear()

        # Reset progress bars and logs
        self.epoch_progress['value'] = 0
        self.gradient_progress['value'] = 0
        self.train_log.config(state='normal')
        self.train_log.delete('1.0', tk.END)
        self.train_log.config(state='disabled')
        self.log_train("Starting training...\n")

        # Initialize time data
        with self.time_lock:
            self.time_data['start_time'] = time.time()
            self.time_data['epochs_done'] = 0
            self.time_data['epochs'] = epochs
            self.time_data['remaining'] = 0

        # Start training in a separate thread
        training_thread = threading.Thread(target=self.training_process, args=(epochs, num_threads), daemon=True)
        training_thread.start()

    def training_process(self, epochs: int, num_threads: int):
        """
        The actual training process running in a separate thread.
        """
        try:
            self.log_train("Training thread started.\n")
            train_model(
                self.model,
                self.X,
                self.Y,
                epochs=epochs,
                lr=self.optimizer.lr,
                num_threads=num_threads,
                log_queue=self.log_queue,
                stop_flag=self.stop_flag,
                time_lock=self.time_lock,
                time_data=self.time_data,
                optimizer=self.optimizer
            )
            if not self.stop_flag.is_set():
                self.log_train("Training completed successfully.\n")
                messagebox.showinfo("Training Completed", "Model training completed successfully.")
        except Exception as e:
            err_msg = f"Training error:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Training Error", err_msg)
        finally:
            # Re-enable buttons after training
            self.train_button.config(state='normal')
            self.save_button.config(state='normal')
            self.load_button.config(state='normal')
            self.infer_button.config(state='normal')
            if not self.stop_flag.is_set():
                self.epoch_progress['value'] = 100
                self.gradient_progress['value'] = 100

            # Update evaluation metrics
            self.evaluate_model()

    def stop_training(self):
        """
        Gracefully stop the training after the current epoch.
        """
        self.stop_flag.set()
        self.log_train("Stop signal sent. Will stop after current epoch.\n")

    def hard_stop(self):
        """
        Immediately terminate the application.
        """
        self.log_train("Hard stop invoked. Terminating immediately.\n")
        os._exit(1)

    def save_model(self):
        """
        Save the trained model and token mappings.
        """
        try:
            save_path = filedialog.asksaveasfilename(title="Save Model", defaultextension=".qelm",
                                                     filetypes=[("QELM Files", "*.qelm", "*.qlm"), ("All Files", "*.*")])
            if save_path:
                self.model.save_model(save_path)
                if self.token_to_id:
                    token_map_path = save_path.replace(".qelm", "_token_map.json")
                    with open(token_map_path, 'w') as f:
                        json.dump(self.token_to_id, f, indent=4)
                    self.log_train(f"Token mappings saved to {token_map_path}\n")
                messagebox.showinfo("Model Saved", f"Model saved to {save_path}")
        except Exception as e:
            err_msg = f"Save model error:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Save Error", err_msg)

    def load_model(self):
        """
        Load a saved model and its token mappings.
        """
        try:
            load_path = filedialog.askopenfilename(title="Load Model",
                                                  filetypes=[("QELM Files", "*.qelm", "*.qlm"), ("All Files", "*.*")])
            if load_path:
                self.model.load_model(load_path)
                token_map_path = load_path.replace(".qelm", "_token_map.json")
                try:
                    with open(token_map_path, 'r') as f:
                        self.token_to_id = json.load(f)
                    self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
                    self.log_train(f"Loaded token mappings from {token_map_path}\n")
                    self.display_token_map()
                except FileNotFoundError:
                    self.log_train("No token mappings file found.\n")
                messagebox.showinfo("Model Loaded", f"Model loaded from {load_path}")
        except Exception as e:
            err_msg = f"Load model error:\n{traceback.format_exc()}"
            self.log_train(err_msg + "\n")
            messagebox.showerror("Load Error", err_msg)

    def run_inference(self):
        """
        Run inference based on the input token.
        """
        input_token = self.input_token_entry.get().strip().lower()
        if not input_token:
            messagebox.showerror("Input Error", "Please enter an input token for inference.")
            return

        try:
            max_length = int(self.max_length_entry.get())
            temperature = float(self.temperature_entry.get())
            if max_length <= 0 or temperature <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter positive values for max length and temperature.")
            return

        self.infer_button.config(state='disabled')
        self.log_infer(f"Running inference for '{input_token}' with max_length={max_length} and temperature={temperature}...\n")

        # Start inference in a separate thread
        inference_thread = threading.Thread(target=self.inference_process, args=(input_token, max_length, temperature), daemon=True)
        inference_thread.start()

    def inference_process(self, input_token: str, max_length: int, temperature: float):
        """
        The actual inference process running in a separate thread.
        """
        try:
            if input_token not in self.token_to_id:
                raise ValueError(f"Input token '{input_token}' not found in token mappings.")
            input_id = self.token_to_id[input_token]
            generated_tokens, response = run_inference(
                self.model,
                [input_id],
                self.token_to_id,
                self.id_to_token,
                max_length=max_length,
                temperature=temperature,
                log_callback=self.log_infer
            )
            messagebox.showinfo("Inference Completed", "Inference completed successfully.")
        except Exception as e:
            err_msg = f"Inference error:\n{traceback.format_exc()}"
            self.log_infer(err_msg + "\n")
            messagebox.showerror("Inference Error", err_msg)
        finally:
            self.infer_button.config(state='normal')

    def load_token_map(self):
        """
        Load token-to-ID mappings from a JSON file.
        """
        try:
            file_path = filedialog.askopenfilename(title="Load Token Map",
                                                   filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
            if file_path:
                with open(file_path, 'r') as f:
                    self.token_to_id = json.load(f)
                self.id_to_token = {int(idx): token for token, idx in self.token_to_id.items()}
                self.log_token_map(f"Loaded token mappings from {file_path}\n")
                self.display_token_map()
                messagebox.showinfo("Token Map Loaded", f"Token mappings loaded from {file_path}")
        except Exception as e:
            err_msg = f"Load token map error:\n{traceback.format_exc()}"
            self.log_token_map(err_msg + "\n")
            messagebox.showerror("Load Error", err_msg)

    def display_token_map(self):
        """
        Display the token-to-ID mappings in the GUI.
        """
        self.token_map_display.config(state='normal')
        self.token_map_display.delete('1.0', tk.END)
        self.token_map_display.insert(tk.END, "Token Mappings (Token: ID):\n\n")
        for token, idx in sorted(self.token_to_id.items(), key=lambda x: x[1]):
            self.token_map_display.insert(tk.END, f"{token}: {idx}\n")
        self.token_map_display.config(state='disabled')

    def update_resource_usage(self):
        """
        Update CPU usage and display GPU status.
        """
        if self.process:
            cpu_usage = f"{self.process.cpu_percent(interval=None)}%"
        else:
            cpu_usage = "psutil not installed"

        self.cpu_label.config(text=f"CPU: {cpu_usage}")
        self.gpu_label.config(text=f"GPU: Check externally (e.g., nvidia-smi)")

        self.master.after(1000, self.update_resource_usage)

    def update_time_label(self):
        """
        Update the elapsed and remaining training time.
        """
        with self.time_lock:
            elapsed = time.time() - self.time_data.get('start_time', 0)
            elapsed_str = f"{elapsed:.1f}s"

            remaining = self.time_data.get('remaining', 0)
            if remaining > 0:
                remaining_str = f"{remaining:.1f}s"
            else:
                remaining_str = "Estimating..."

        self.time_label.config(text=f"Elapsed: {elapsed_str} | Remaining: {remaining_str}")

        # Update every second
        self.master.after(1000, self.update_time_label)

    def evaluate_model(self):
        """
        Evaluate the model using Perplexity and BLEU score.
        """
        # Compute Perplexity on training data
        perplexities = []
        for x, y in zip(self.X, self.Y):
            logits = self.model.forward([x], use_residual=True)
            perp = perplexity(logits, y)
            perplexities.append(perp)
        avg_perplexity = np.mean(perplexities)

        # Compute BLEU score (requires reference and hypothesis; simplistic implementation)
        # Here, we treat training data as reference and model's predictions as hypothesis
        # This is not standard but serves demonstration purposes
        hypotheses = []
        references = []
        for x, y in zip(self.X, self.Y):
            logits = self.model.forward([x], use_residual=True)
            predicted = np.argmax(logits)
            hypotheses.append([self.id_to_token.get(predicted, "<UNK>")])
            references.append([self.id_to_token.get(y, "<UNK>")])

        bleu_scores = []
        for ref, hyp in zip(references, hypotheses):
            bleu = bleu_score(ref, hyp)
            bleu_scores.append(bleu)
        avg_bleu = np.mean(bleu_scores)

        # Update GUI labels
        self.perplexity_label.config(text=f"Perplexity: {avg_perplexity:.4f}")
        self.bleu_label.config(text=f"BLEU Score: {avg_bleu:.4f}")

def main():
    """
    Entry point for the application.
    """
    try:
        root = tk.Tk()
        gui = QELM_GUI(root)
        root.mainloop()
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.critical(f"Unexpected error:\n{error_trace}")
        # Create a hidden root to display the message box
        hidden_root = tk.Tk()
        hidden_root.withdraw()
        messagebox.showerror("Unexpected Error", f"An unexpected error occurred:\n{e}\n\nCheck the log file for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
