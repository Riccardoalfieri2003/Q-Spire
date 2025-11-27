# QSpire - Quantum Code Smell Detector

QSpire is a VS Code extension designed to detect Quantum Code Smells in the Qiskit Framework for Python. As Quantum Computing is a novel and rapidly evolving field, working on Quantum Computing Engineering practices such as smell detection can significantly improve code maintainability and quality.

## Features

- **Dual Analysis Modes**: Perform both Static and Dynamic analysis on your quantum code
  - **Dynamic Analysis**: Executes the user's code to detect smells
  - **Static Analysis**: Simulates the user's code 
- **Visual Feedback**: Intuitive GUI with real-time highlighting of detected code smells
- **Batch Processing**: CLI support for analyzing entire folders
- **AI-Powered Explanations**: Integrated LLM feature to get explanations and resolutions for detected smells
- **Customizable Thresholds**: Configure detection sensitivity for each smell type

## Quantum Code Smells Detected

QSpire currently detects 8 different quantum code smells:

- **CG** - Use of Customized Gates
- **LPQ** - No-alignment between the Logical and Physical Qubits
- **IM** - Intermediate Measurements
- **IQ** - Initialization of Qubits differently from |0>
- **IdQ** - Idle Qubit
- **NC** - Non-parameterized Circuit
- **ROC** - Repeated set of Operations on Circuit
- **LC** - Long Circuit

These smells are based on empirical research in quantum software engineering.

## Installation

# Installation

## Prerequisites

Before installing QSpire, ensure you have the following:

* **Python**: Version 3.8 to 3.13 (3.8 or higher, but less than 3.14)
* **VS Code**: Version 1.60.0 or higher
* **Cloned QSpire repository**: The project must be opened in VS Code

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Riccardoalfieri2003/Q-Spire
```

### 2. Open in VS Code

* Open VS Code
* Open the cloned QSpire project folder

### 3. Install CLI and Necessary Libraries

* Open the terminal in VS Code (`Terminal → New Terminal`)
* Run the following command in the terminal:

```bash
pip install -e .
```

This installs the QSpire CLI tool and all required dependencies.

### 4. Install GUI Extension

* In VS Code, press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) to open the Command Palette
* Type `>Extensions: Install from VSIX...`
* Select the `qspire-0.0.1.vsix` file from the project directory

### 5. Final Step

* Restart VS Code if prompted to complete the installation

## Verification

After installation, you should be able to:

* Use the `qspire` command in your terminal
* Access QSpire commands through VS Code's Command Palette

## Usage

### GUI Mode

1. Open a Python file containing Qiskit code
2. Click the circuit button on the top right, next to the "Play" button
3. The QSpire GUI will open with two main sections:

**Detection Window:**
- Select the file to analyze (automatically detects the current file)
- Choose analysis method: Static or Dynamic
- Click "Analyze" to start detection

**Results Window:**
- View detected smells with highlighting in your code
- Hover over highlighted sections for smell details
- Configure API Key and Model settings for AI-powered explanations
- Click on individual smells to get detailed explanations and suggested fixes

### CLI Mode

Basic command structure:

```bash
qspire -method <static|dynamic> <resource_path> [output_folder]
```

**Parameters:**
- `method`: Required. Either `static` or `dynamic`
- `resource_path`: Required. Path to a file or folder to analyze
- `output_folder`: Optional. Directory where results will be saved

**Examples:**

Analyze a single file with dynamic analysis:
```bash
qspire -dynamic "C:/my_quantum_code.py"
```

Analyze an entire folder with static analysis and save results:
```bash
qspire -static "C:/quantum_project" "C:/results"
```

**Absolute path** is needed for both *resource* and *output_folder*

## Configuration

### Detection Thresholds

Customize smell detection sensitivity by modifying thresholds in the configuration. Access settings through:
- GUI: Settings panel in the Results window
- CLI: Edit the configuration file (config.json)

### AI Explanations

Configure the LLM integration for smell explanations:

1. **API Key**: Enter your OpenRouter API key
2. **Model Selection**: Default model is `deepseek/deepseek-r1-distill-llama-70b:free`
   - You can select any compatible model from ([OpenRouter](https://openrouter.ai/))

## Architecture

QSpire features a modular architecture:

```
qspire/
├── smells/              # Smell definitions and detectors
│   ├── CG/             # Circuit Granularity smell
│   ├── LPQ/            # Long Parameter Queue smell
│   ├── IM/             # Improper Measurement smell
│   └── ...             # Other smell modules
├── detection/          # Analysis engines
│   ├── static/         # Static analysis
│   └── dynamic/        # Dynamic analysis
...
```

Each smell module contains:
- **Definition**: Formal description of the smell
- **Detector**: Logic to identify the smell
- **Explainer**: AI-powered explanation generator via OpenRouter API

## Example Output

Example smell instances after detection:

```json
{
  "type": "IM",
  "row": 17,
  "column_start": 9,
  "column_end": 28,
  "explanation": "",
  "suggestion": "",
  "circuit_name": "qc",
  "qubit": 0
}
```

```json
{
  "type": "LPQ",
  "row": 48,
  "column_start": 18,
  "column_end": 45,
  "circuit_name": "circuits"
}
```

```json
{
  "type": "CG",
  "row": 26,
  "column_start": 1,
  "column_end": 46,
  "circuit_name": "qc2",
  "matrix": "<Call to eye>"
}
```

## Contributing

Thanks to QSpire's modular architecture, adding support for new quantum code smells is straightforward:

1. Create a new smell module in the `smells/` directory
2. Implement the detector logic following existing patterns
3. Add an explainer for AI-powered insights
4. Register the smell in the detection engine

We welcome contributions from the quantum computing community!

## Acknowledgments

The quantum code smells detected by QSpire are based on the research paper:

**"The Smelly Eight: An Empirical Study on the Prevalence of Code Smells in Quantum Computing"** by Chen et al.: [paper](https://dl.acm.org/doi/10.1109/ICSE48619.2023.00041)

This tool aims to bring empirically-validated software engineering practices to the quantum computing domain.


## Support

For issues, questions, or contributions, please contact ric.alfa.2003@gmail.com via gmail.

---

**Note**: Quantum computing is an emerging field. As new code smells are identified and defined by the research community, QSpire will be updated to detect them.


## Credits

- Extension icon created by Freepik - Flaticon. [quantum-computing icons](https://www.flaticon.com/free-icons/quantum-computing)