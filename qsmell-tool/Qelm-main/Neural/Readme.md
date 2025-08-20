# Quantum-Enhanced Language Model (QELM) – Neuron

Welcome to the **Quantum-Enhanced Language Model (QELM) Neuron** repository! This project leverages quantum computing and neural
network–inspired architectures to explore advanced language processing using **Qiskit** and quantum hardware simulators.

***Update — 2025‑07‑27***

Preparing a Neural‑focused update. Core addition: a gradient‑flow pipeline implementing online, quantum‑assisted optimization. Gradients are computed via QELM‑derived flows with an enforced sub‑bit gradient regime. The pipeline establishes a branching quantum channel: state information is duplicated at the channel level prior to observation; one branch is measured, while the companion branch maintains coherence (or a learned surrogate) to continue parameter updates without full state re‑preparation. This yields a self‑reinforcing quantum flow whose effective learning rate and search dynamics increase with experience. Target direction: QELM as a self‑learning QPU architecture — a quantum “brain.”


## 🚀 Project Overview

QELM Neuron is a cutting-edge framework designed to train and simulate quantum-enhanced language models by integrating the following components:

1. **Quantum Neuron (20-Qubit Logical Unit)**:
   - Logical qubit encoding with error correction.
   - Parameterized gates for efficient quantum processing.

2. **Graphene-Like Layer Structure**:
   - Multi-neuron hexagonal lattice inspired by graphene.
   - Entangled neurons for brain-like connectivity.

3. **Stacked Quantum Layers**:
   - Deep architectures for scalable language models.
   - Inter-layer entanglement for enhanced quantum computations.

4. **Hybrid Quantum-Classical Training**:
   - Classical embedding and output layers.
   - Quantum neural networks for intermediate computations.

---

## 🔑 Key Features

- **Quantum Neuron Implementation**: Logical qubits with error correction for robust computations.
- **Scalable Architectures**: Stacked graphene-inspired layers for multi-layer neural networks.
- **Quantum-Classical Hybrid**: Combines classical embedding and output layers with quantum computations.
- **Simulation Support**: Utilize Qiskit Aer to simulate noise and quantum state evolution.
- **Customizable Parameters**: Trainable rotation angles, embeddings, and weights.
- **Real and Synthetic Dataset Support**: Create or load datasets for training and evaluation.

---

## 📂 Repository Structure

```
QELM/
├── Neuron.py            # Main QELM script with all core implementations.
├── README.md            # Project documentation.
├── requirements.txt     # Required Python libraries.
└── quantum_llm.qelm     # Example trained model file (generated after training).
```

---

## 🛠️ Requirements

To run the project, ensure the following Python dependencies are installed:

- `qiskit`
- `qiskit-aer`
- `numpy`
- `scipy`
- `nltk`
- `tensorflow`
- `psutil`


Install all dependencies using:
```bash
pip install -r requirements.txt
```

---

## 📖 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/R-D-BioTech-Alaska/Qelm/tree/main/Neural/Neuron.git
   cd Neuron
   ```

2. Run the main script:
   ```bash
   python Neuron.py
   ```

3. During execution, the script will:
   - Initialize the QELM model with reduced parameters for simulation.
   - Train the model using synthetic datasets.
   - Save the trained model (`quantum_llm.qelm`).
   - Evaluate the model for fidelity and predictions.

4. Customize model parameters, dataset paths, or training settings directly in the script.

---

## 📊 Example Output

After running the script, you may see logs like the following:

```plaintext
2024-12-15 12:00:00 - INFO - Starting Epoch 1/2
2024-12-15 12:00:00 - INFO - Epoch 1/2, Loss: 0.024567
2024-12-15 12:00:00 - INFO - Model successfully saved to quantum_llm.qelm
Sample Fidelity: 0.9234
Sample Fidelity After Loading: 0.9210
```

---

## 📈 Training and Evaluation

- **Synthetic Dataset**:
  Generate a dataset with:
  ```python
  X, Y = create_synthetic_dataset(vocab_size=16, num_samples=100)
  ```

- **Real Dataset**:
  Load a real dataset from a text file:
  ```python
  X, Y, token_to_id = load_real_dataset(file_path="data.txt", vocab_size=5000)
  ```

- **Train the Model**:
  Update `epochs` and `learning_rate` for your needs in:
  ```python
  train_quantum_model(qlm, X, Y, epochs=10, learning_rate=0.01)
  ```

---

## 📚 Documentation

For a detailed explanation of the model architecture and quantum principles, check the inline docstrings in `Neuron.py`.

---

## 🤝 Contributing

We welcome contributions! Feel free to fork this repository, submit issues, or create pull requests.

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments

Special thanks to the Qiskit community and all contributors for advancing quantum computing and machine learning.

---
