```markdown
# 🔮 Mini-Transformer Language Model (GPT-like) for Text Generation

A minimal, educational implementation of a Transformer-based language model (GPT-style) using PyTorch. This model learns to generate text character-by-character from a custom dataset (in this case, a subset of the *Harry Potter* books).

> 📚 Inspired by the structure of nanoGPT and Karpathy’s tutorials, this repo aims to balance simplicity, readability, and effectiveness.

---

## 🚀 Features

- Transformer architecture with:
  - Multi-head self-attention
  - Causal (masked) attention mechanism
  - Position and token embeddings
  - LayerNorm and Feedforward blocks
- Character-level tokenization
- Text generation with temperature-based sampling
- Configurable hyperparameters
- Training and evaluation on a CPU or GPU

---

## 🗂 Dataset

The model uses a text file (e.g. `harrypotter.txt`) containing raw English text.

### 🔧 Prepare your dataset:

1. Place your `.txt` file inside the project directory.
2. Update the path in the code:
   ```python
   with open('path/to/your_file.txt', 'r', encoding='utf-8') as f:
       text = f.read()
   ```

---

## 🛠 Requirements

- Python 3.7+
- PyTorch (GPU recommended for training)

Install dependencies:

```bash
pip install torch
```

---

## 🏋️‍♂️ Training

You can start training by running the script directly:

```bash
python bigram.py
```

> Training parameters like `batch_size`, `n_embed`, `block_size`, and `max_iters` can be modified at the top of the script.

The script will:
- Load and tokenize the dataset
- Train the model using a simple loop
- Print loss values every `eval_interval` steps
- Generate sample text at the end

---

## ✨ Text Generation

You can generate new text after training using:

```python
print(generate_text("The sky was dark", max_new_tokens=300))
```

This will output character-by-character generated text conditioned on your prompt.

---

## 📊 Sample Output

```text
The sky was dark and the wind howled through the trees. Harry shivered, pulling his cloak tighter around him as he crept...
```

(Note: Output will vary depending on dataset and training time.)

---

## 🧠 Customization Ideas

- Use a word-level tokenizer instead of character-level
- Add temperature or top-k sampling for generation
- Train on a larger dataset
- Use a checkpointing system to save/load models
- Add attention visualization for interpretability

---

## 📁 File Structure

```
.
├── bigram.py            # Main training and generation script
├── harrypotter.txt      # Your dataset file (or replace with your own)
└── README.md            # This file
```

---

## 🙌 Acknowledgments

- https://github.com/Infatoshi for his amazing educational content
- [Andrej Karpathy](https://github.com/karpathy) for his amazing educational content
- PyTorch team for making deep learning accessible
```

---
