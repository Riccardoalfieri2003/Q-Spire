#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to Generate Vocabulary from a Large Corpus or Dataset and Update a Model Qelm/JSON with a Tkinter GUI.
I have included multiple different tokenization methods, limiting this might work better.
This is still rudi for just incase you need a new vocab for qelm
"""

import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from collections import Counter
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


def build_subword_tokenizer(corpus_tokens, vocab_size, special_tokens=None):

    # Start with each token as an individual 'symbol'
    token_freq = Counter(corpus_tokens)

    # If special tokens are provided, reserve them at the start of the vocabulary
    if special_tokens is None:
        special_tokens = []

    # Convert tokens into a list of characters (sub-tokens)
    subword_freq = Counter()
    for token, freq in token_freq.items():
        chars = tuple(list(token))  # store as a tuple to be hashable
        subword_freq[chars] += freq

    # Merging procedure
    def get_most_frequent_pair(freq_dict):
        pairs = Counter()
        for seq, freq in freq_dict.items():
            if len(seq) < 2:
                continue
            # Count adjacent pairs
            for i in range(len(seq) - 1):
                pairs[(seq[i], seq[i+1])] += freq
        if not pairs:
            return None
        return pairs.most_common(1)[0][0]  # return the pair with the highest freq

    # Initialize a merges list which will store merges performed
    merges = []
    current_vocab_size = len(subword_freq)

    max_merges = vocab_size - len(special_tokens)

    while current_vocab_size < max_merges:
        pair_to_merge = get_most_frequent_pair(subword_freq)
        if not pair_to_merge:
            break
        merges.append(pair_to_merge)
        # Merge the pair in subword_freq
        new_subword_freq = Counter()
        for seq, freq in subword_freq.items():
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == pair_to_merge[0] and seq[i+1] == pair_to_merge[1]:
                    # Merge into a single subword
                    new_seq.append(''.join(pair_to_merge))
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            new_subword_freq[tuple(new_seq)] += freq
        subword_freq = new_subword_freq
        current_vocab_size += 1

    # e.g., if we ended with subword_freq containing ('qu', 'an', 't', 'um'), we have these subwords
    unique_symbols = set()
    for seq in subword_freq:
        unique_symbols.update(seq)

    # Insert special tokens first
    token_to_id = {}
    idx_counter = 0
    for sp_token in special_tokens:
        token_to_id[sp_token] = idx_counter
        idx_counter += 1

    for sym in sorted(unique_symbols):
        # Skip if it's already a special token
        if sym not in token_to_id:
            token_to_id[sym] = idx_counter
            idx_counter += 1

    return token_to_id


def tokenize_bpe_like(text):

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    return tokens


def generate_vocabulary_from_corpus(
    corpus: str, 
    vocab_size: int, 
    tokenization_method: str = 'basic', 
    use_lemmatization: bool = False
) -> dict:

    if not corpus.strip():
        raise ValueError("Input corpus is empty or contains no valid text.")

    # Some standard special tokens; can be expanded as needed.
    SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    if tokenization_method == 'basic':
        # Basic tokenization (NLTK word_tokenize)
        tokens = word_tokenize(corpus.lower())
    elif tokenization_method == 'bpe':
        # BPE-like tokenization (two-step: naive word split + subword merges)
        tokens = tokenize_bpe_like(corpus)
    elif tokenization_method == 'char':
        # Character-based tokenization
        tokens = list(corpus.replace(" ", "_"))  # replace spaces with underscores to preserve them
    else:
        raise ValueError(f"Unsupported tokenization method: {tokenization_method}")

    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        # For BPE-like or char-based, lemmatization doesn't always make sense.
        if tokenization_method in ['basic', 'bpe']:
            new_tokens = []
            for tk in tokens:
                if tk.isalpha():
                    new_tokens.append(lemmatizer.lemmatize(tk))
                else:
                    new_tokens.append(tk)
            tokens = new_tokens

    if tokenization_method == 'bpe':
        token_to_id = build_subword_tokenizer(tokens, vocab_size, special_tokens=SPECIAL_TOKENS)
        return token_to_id
    else:
        if not tokens:
            raise ValueError("Tokenization produced no tokens. Ensure the input corpus contains valid text.")

        token_freq = Counter(tokens)

        # Insert special tokens with high frequency to ensure they appear
        for sp_tok in SPECIAL_TOKENS:
            token_freq[sp_tok] = 999999999

        # Take the most common tokens
        most_common_tokens = [token for token, _ in token_freq.most_common(vocab_size)]
        token_to_id = {token: idx for idx, token in enumerate(most_common_tokens)}
        return token_to_id


def update_model_with_vocabulary(json_file: str, token_to_id: dict):
    """
    Update the model JSON file with a `token_to_id` mapping and ensure embeddings match the vocabulary.
    """
    # Load the model JSON file
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Model JSON file not found: {json_file}")

    with open(json_file, "r") as f:
        model_dict = json.load(f)

    # Update the model with token mappings
    model_dict["token_to_id"] = token_to_id
    vocab_size = len(token_to_id)

    # Ensure embeddings match the new vocabulary size
    if "embeddings" in model_dict:
        embed_dim = len(model_dict["embeddings"][0])
        model_dict["embeddings"] = (np.random.randn(vocab_size, embed_dim) * 0.01).tolist()
    else:
        raise ValueError("Embeddings are missing in the model JSON.")

    # Save the updated model JSON file
    with open(json_file, "w") as f:
        json.dump(model_dict, f, indent=4)

    print(f"Model updated successfully with vocabulary. Saved to: {json_file}")


def validate_input_text(text: str, token_to_id: dict, tokenization_method: str = 'basic'):

    if "[UNK]" not in token_to_id:
        raise ValueError("Vocabulary missing the [UNK] special token. Cannot handle out-of-vocab tokens properly.")

    text = text.lower()
    
    if tokenization_method == 'basic':
        tokens = word_tokenize(text)
    elif tokenization_method == 'bpe':
        # Check each subword element against the vocab; if not present, we label it as [UNK].
        tokens = tokenize_bpe_like(text)
    elif tokenization_method == 'char':
        tokens = list(text.replace(" ", "_"))
    else:
        tokens = word_tokenize(text)

    valid_tokens = []
    missing_tokens = []

    for tk in tokens:
        if tk in token_to_id:
            valid_tokens.append(tk)
        else:
            missing_tokens.append(tk)

    if not valid_tokens and missing_tokens:
        raise ValueError(
            f"Input text contains tokens that are not in the vocabulary. Missing tokens: {missing_tokens}. "
            f"Consider updating or expanding the corpus to include these words."
        )

    return valid_tokens


class VocabularyGUI:
    def __init__(self, master):
        self.master = master
        master.title("Advanced Vocabulary Generator and Model Updater")

        # Initialize variables
        self.json_file = tk.StringVar()
        self.vocab_size = tk.IntVar(value=100000)
        self.sample_input = tk.StringVar()
        self.corpus_mode = tk.StringVar(value="manual")
        self.dataset_path = tk.StringVar()
        self.tokenization_method = tk.StringVar(value='basic')
        self.use_lemmatization = tk.BooleanVar(value=False)

        # Layout configuration
        self.create_widgets()

    def create_widgets(self):
        # Mode Selection
        mode_frame = tk.Frame(self.master)
        mode_frame.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(mode_frame, text="Corpus Source:").pack(anchor=tk.W)
        modes = [("Manual Text Corpus", "manual"), ("Use Dataset", "dataset")]
        for text, mode in modes:
            tk.Radiobutton(mode_frame, text=text, variable=self.corpus_mode, value=mode, command=self.toggle_corpus_source).pack(anchor=tk.W)

        # Dataset Selection
        self.dataset_frame = tk.Frame(self.master)
        self.dataset_frame.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(self.dataset_frame, text="Dataset File/Directory:").pack(side=tk.LEFT)
        self.dataset_entry = tk.Entry(self.dataset_frame, textvariable=self.dataset_path, width=50, state='disabled')
        self.dataset_entry.pack(side=tk.LEFT, padx=5)
        self.browse_dataset_button = tk.Button(self.dataset_frame, text="Browse", command=self.browse_dataset, state='disabled')
        self.browse_dataset_button.pack(side=tk.LEFT)

        # JSON File Selection
        json_frame = tk.Frame(self.master)
        json_frame.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(json_frame, text="Model JSON File:").pack(side=tk.LEFT)
        self.json_entry = tk.Entry(json_frame, textvariable=self.json_file, width=50)
        self.json_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(json_frame, text="Browse", command=self.browse_json).pack(side=tk.LEFT)

        # Vocabulary Size
        vocab_frame = tk.Frame(self.master)
        vocab_frame.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(vocab_frame, text="Vocabulary Size:").pack(side=tk.LEFT)
        self.vocab_entry = tk.Entry(vocab_frame, textvariable=self.vocab_size, width=10)
        self.vocab_entry.pack(side=tk.LEFT, padx=5)

        # Tokenization Method
        tok_method_frame = tk.LabelFrame(self.master, text="Tokenization Method")
        tok_method_frame.pack(padx=10, pady=5, fill=tk.X)

        methods = [
            ("Basic (NLTK word_tokenize)", "basic"),
            ("BPE-like (Subword)", "bpe"),
            ("Character-based", "char")
        ]
        for mtext, mval in methods:
            tk.Radiobutton(tok_method_frame, text=mtext, variable=self.tokenization_method, value=mval).pack(anchor=tk.W)

        # Lemmatization Toggle
        lemma_frame = tk.Frame(self.master)
        lemma_frame.pack(padx=10, pady=5, fill=tk.X)

        tk.Checkbutton(lemma_frame, text="Use Lemmatization", variable=self.use_lemmatization).pack(anchor=tk.W)

        # Text Corpus
        self.corpus_frame = tk.Frame(self.master)
        self.corpus_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        tk.Label(self.corpus_frame, text="Text Corpus:").pack(anchor=tk.W)
        self.corpus_text = scrolledtext.ScrolledText(self.corpus_frame, height=10)
        self.corpus_text.pack(fill=tk.BOTH, expand=True)

        # Change the default corpus anyway you want, just ensure it includes a full alphabet
        default_corpus = """
Quantum computing is an area of computing focused on developing computer technology
based on the principles of quantum theory. Language models aim to generate meaningful
text based on input, bridging the gap between technology and human interaction.

Great literature includes works like those of Shakespeare, Dickens, and Austen.
Scientific advancements span physics, biology, and chemistry. Everyday conversations 
include phrases like "hello," "how are you?" and "what's the weather like today?"

Advanced topics in artificial intelligence, machine learning, and deep learning
pave the way for innovation. Historical texts, scientific papers, and novels 
contribute to a rich corpus for modeling.

Popular culture references, news headlines, and casual dialogue make up modern 
conversations. The integration of complex fields like robotics and autonomous 
systems revolutionizes industries. Let's ensure our corpus is broad enough to 
capture all nuances of human language!
"""
        self.corpus_text.insert(tk.END, default_corpus.strip())

        # Sample Input
        sample_frame = tk.Frame(self.master)
        sample_frame.pack(padx=10, pady=5, fill=tk.X)

        tk.Label(sample_frame, text="Sample Input:").pack(side=tk.LEFT)
        self.sample_entry = tk.Entry(sample_frame, textvariable=self.sample_input, width=50)
        self.sample_entry.pack(side=tk.LEFT, padx=5)
        self.sample_entry.insert(0, "hello world, what's quantum computing?")

        # Buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(padx=10, pady=10)

        tk.Button(button_frame, text="Generate Vocabulary", command=self.generate_vocabulary).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Validate Input", command=self.validate_input).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Update Model JSON", command=self.update_model).pack(side=tk.LEFT, padx=5)

        # Log Area
        log_frame = tk.Frame(self.master)
        log_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

        tk.Label(log_frame, text="Log:").pack(anchor=tk.W)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def toggle_corpus_source(self):
        mode = self.corpus_mode.get()
        if mode == "manual":
            self.dataset_entry.config(state='disabled')
            self.browse_dataset_button.config(state='disabled')
            self.corpus_text.config(state='normal')
            self.log("Switched to Manual Text Corpus mode.")
        else:
            self.dataset_entry.config(state='normal')
            self.browse_dataset_button.config(state='normal')
            self.corpus_text.config(state='disabled')
            self.log("Switched to Dataset mode.")

    def browse_json(self):
        filepath = filedialog.askopenfilename(
            title="Select Model JSON File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if filepath:
            self.json_file.set(filepath)
            self.log(f"Selected JSON file: {filepath}")

    def browse_dataset(self):
        filepath = filedialog.askopenfilename(
            title="Select Dataset File or Directory",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if not filepath:
            # If no file is selected, try selecting a directory
            filepath = filedialog.askdirectory(title="Select Dataset Directory")
        if filepath:
            self.dataset_path.set(filepath)
            self.log(f"Selected dataset path: {filepath}")

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def load_corpus_from_dataset(self, path):
        if os.path.isfile(path):
            if not path.lower().endswith('.txt'):
                raise ValueError("Selected file is not a text file (*.txt).")
            with open(path, 'r', encoding='utf-8') as f:
                corpus = f.read()
            self.log(f"Loaded corpus from file: {path}")
            return corpus
        elif os.path.isdir(path):
            corpus = ""
            txt_files = [f for f in os.listdir(path) if f.lower().endswith('.txt')]
            if not txt_files:
                raise ValueError("No text files (*.txt) found in the selected directory.")
            for file in txt_files:
                file_path = os.path.join(path, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    corpus += f.read() + " "
                self.log(f"Loaded text from: {file_path}")
            return corpus
        else:
            raise ValueError("Selected path is neither a file nor a directory.")

    def generate_vocabulary(self):
        self.log("Generating vocabulary...")
        mode = self.corpus_mode.get()
        vocab_size = self.vocab_size.get()
        tok_method = self.tokenization_method.get()
        lemma_flag = self.use_lemmatization.get()

        try:
            if mode == "manual":
                corpus = self.corpus_text.get("1.0", tk.END)
            else:
                dataset_path = self.dataset_path.get()
                if not dataset_path:
                    raise ValueError("Dataset path is not specified.")
                corpus = self.load_corpus_from_dataset(dataset_path)

            # Generate the vocabulary with advanced options
            token_to_id = generate_vocabulary_from_corpus(
                corpus,
                vocab_size,
                tokenization_method=tok_method,
                use_lemmatization=lemma_flag
            )
            self.log(f"Generated {len(token_to_id)} tokens using {tok_method} tokenization. Lemmatization={lemma_flag}.")
            messagebox.showinfo("Success", f"Vocabulary generated with {len(token_to_id)} tokens.")
            self.generated_token_to_id = token_to_id  # Store for later use
        except ValueError as e:
            self.log(f"Error while generating vocabulary: {e}")
            messagebox.showerror("Error", f"Failed to generate vocabulary:\n{e}")

    def validate_input(self):
        self.log("Validating sample input...")
        sample = self.sample_input.get()
        tok_method = self.tokenization_method.get()
        try:
            if not hasattr(self, 'generated_token_to_id'):
                raise ValueError("Vocabulary not generated yet. Please generate the vocabulary first.")
            valid_tokens = validate_input_text(sample, self.generated_token_to_id, tokenization_method=tok_method)
            self.log(f"Valid tokens: {valid_tokens}")
            messagebox.showinfo("Validation Success", f"Valid tokens:\n{valid_tokens}")
        except ValueError as e:
            self.log(f"Error while validating input: {e}")
            messagebox.showerror("Validation Error", f"Failed to validate input:\n{e}")

    def update_model(self):
        self.log("Updating model JSON with vocabulary...")
        json_file = self.json_file.get()
        try:
            if not hasattr(self, 'generated_token_to_id'):
                raise ValueError("Vocabulary not generated yet. Please generate the vocabulary first.")
            if not json_file:
                raise ValueError("JSON file path is not specified.")
            update_model_with_vocabulary(json_file, self.generated_token_to_id)
            self.log("Vocabulary update complete.")
            messagebox.showinfo("Success", "Model JSON updated successfully.")
        except (FileNotFoundError, ValueError) as e:
            self.log(f"Error while updating model JSON: {e}")
            messagebox.showerror("Update Error", f"Failed to update model JSON:\n{e}")

def main():
    root = tk.Tk()
    gui = VocabularyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
