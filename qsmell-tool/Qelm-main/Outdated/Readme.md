***Outdated but still functioning versions.***

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20the%20Server-blue?style=for-the-badge)](https://discord.gg/sr9QBj3k36)

# Quantum-Enhanced Language Model 

### Using QelmGUI.py
  QelmGUI is pretty self explanatory and more of a comparison model for the original framework.

### Training & Inference with Qelm2.py (Original CMD interface)
Run the unified training and inference script with customizable flags:

#### Basic Command Line Training
```bash
python Qelm2.py --train \
                --dataset /path/to/data.txt \
                --vocab_size 8000 \
                --embed_dim 256 \
                --num_heads 4 \
                --hidden_dim 512 \
                --epochs 5 \
                --lr 0.001
```
**Flags:**
- `--train` : Activate training mode
- `--dataset` : Path to your dataset (.txt)
- `--vocab_size` : Maximum vocabulary size
- `--embed_dim` : Embedding dimension (must be divisible by `--num_heads`)
- `--num_heads` : Number of attention heads
- `--hidden_dim` : Hidden dimension for the feed-forward layers
- `--epochs` : Number of training epochs
- `--lr` : Learning rate

#### Performing Inference
```bash
python Qelm2.py --inference \
                --input_token "hello" \
                --max_length 50 \
                --temperature 1.0 \
                --model /path/to/saved_model.qelm
```
**Flags:**
- `--inference` : Activate inference mode
- `--input_token` : Starting token (word)
- `--max_length` : Maximum number of tokens to generate
- `--temperature` : Sampling temperature (higher values yield more randomness)
- `--model` : Path to a saved `.qelm` model checkpoint

#### Advanced Options
- `--num_blocks N` : Use multi-block quantum transformers (default: 1)
- `--use_advanced_ansatz` : Enable advanced quantum gate configurations
- `--use_data_reuploading` : Enable data reuploading technique
- `--sim_method [cpu|gpu|both|simulation]` : Choose the simulation approach
- `--threads N` : Set the number of threads for parameter-shift gradient computations
- `--decimal_precision N` : Set rounding precision in quantum channel encoding
- `--use_subbit_encoding` : Enable sub-bit encoding for enhanced quantum representation

See all available flags:
```bash
python Qelm2.py --help
```

---
