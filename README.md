# Harry Potter Style Novel Generator

This repository implements a character-level GPT language model trained on Harry Potter text to generate novel-style chapters and scenes in the style of J.K. Rowling's writing. The model is built from scratch using PyTorch, featuring multi-head attention, feedforward layers, and advanced sampling techniques like top-k and top-p (nucleus) sampling.

## Features
- **Character-Level Tokenization**: Simple and efficient tokenizer based on unique characters in the dataset.
- **GPT Architecture**: Includes positional embeddings, transformer blocks with self-attention and feedforward networks, layer normalization, and dropout.
- **Training with Checkpoints**: Supports resuming training from saved model, optimizer, and scheduler states.
- **Interactive Generation**: Command-line interface for generating full chapters, scenes, pages, or custom-length text with adjustable parameters like temperature.
- **Dataset**: Trained on `harry.txt` (assumed to contain concatenated Harry Potter books or excerpts).

Model hyperparameters:
- Embedding dimension: 512
- Number of heads: 8
- Number of layers: 8
- Block size (context length): 512
- Dropout: 0.2
- Total parameters: ~42M (depending on vocab size)

## Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA-enabled GPU recommended for training

Install dependencies:
```
pip install torch
```

## Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```
2. Prepare the dataset: Place your Harry Potter text in `harry.txt` (or modify the code to use your own dataset).

## Training
Run the training script to train the model from scratch or resume from checkpoints.

```bash
python train.py
```

- Training loop runs for 100,000 iterations by default.
- Evaluates loss every 1,000 steps.
- Saves checkpoints (`model.pth`, `optimizer.pth`, `scheduler.pth`) periodically.
- Uses AdamW optimizer with cosine annealing scheduler and gradient clipping.

Example output during training:
```
85  # Vocab size
42.00M parameters
step 0: train loss 4.3921, val loss 4.3918
Checkpoint saved at step 0.
...
```

## Usage: Generation
Use the interactive generator to create Harry Potter-style text.

```bash
python infrence.py
```

Interactive commands:
- `chapter`: Generate a full chapter (~3,750 words / 5,000 tokens).
- `scene`: Generate a scene (~750 words / 1,000 tokens).
- `page`: Generate a page (~250 words / 300 tokens).
- `custom`: Specify custom length and temperature.
- `exit`: Quit the program.
- Or type any prompt directly for quick 500-token generation.

Example session:
```
HARRY POTTER STYLE NOVEL GENERATOR
============================================================

Commands:
  'chapter' - Generate a full chapter (~3750 words)
  'scene' - Generate a scene (~750 words)
  'page' - Generate a page (~250 words)
  'custom' - Custom length generation
  'exit' - Quit

Or just type a prompt to generate a response.

You: chapter
Enter chapter opening (e.g., 'Chapter 12: The Hidden Chamber'): Chapter 12: The Hidden Chamber
Generating 5000 tokens (approximately 3750 words)...
[Generated text appears here...]

Save to file? (y/n): y
Filename: chapter12
Saved to chapter12.txt
```

Generation parameters (editable in code):
- Temperature: 0.85 (controls randomness)
- Top-k: 50 (limits sampling to top k tokens)
- Top-p: 0.92 (nucleus sampling threshold)

## Example Generated Text
Prompt: "Harry woke up in the Gryffindor common room"

Generated (snippet):
```
Harry woke up in the Gryffindor common room, feeling rather disoriented. The fire was crackling merrily in the grate, and a few early risers were scattered about, poring over books or chatting quietly. He rubbed his eyes, wondering why his scar was prickling again...
[Continued story in Harry Potter style]
```

## Contributing
Feel free to fork and submit pull requests for improvements, such as:
- Adding support for larger datasets or fine-tuning.
- Implementing additional sampling methods.
- Optimizing for inference speed.

## License
MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments
- Inspired by Andrej Karpathy's nanoGPT tutorial.
- Dataset: Public domain Harry Potter text.
