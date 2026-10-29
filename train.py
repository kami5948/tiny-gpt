#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================
# Tiny GPT-style toy language model
# ============================================================

CHARSET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,!?\n'-_:;()/[]{}&@#$%*+=<>|~"
)

stoi = {ch: i for i, ch in enumerate(CHARSET)}
itos = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(CHARSET)

# Use a consistent training context length
BLOCK_SIZE = 64

# ============================================================
# Dataset and prompt content
# ============================================================

PROMPT_RESPONSES = [
    ("hello", "hello there friend, i am a tiny language model ready to help"),
    ("hi", "hi! i am a compact ai model designed to learn and generate text"),
    ("how are you", "i am doing well and learning from patterns in the text i see"),
    ("who are you", "i am a tiny gpt-like model built for learning and experimentation"),
    ("what are you", "i am a small transformer model trained to predict the next token"),
    ("what is ai", "ai is the study of creating systems that learn patterns from data"),
    ("what is a model", "a model is a function that learns patterns from examples"),
    ("what is python", "python is a programming language commonly used for ai and data work"),
    ("what is github actions", "github actions lets you automate tasks such as training a model on push"),
    ("what is machine learning", "machine learning is the process of teaching a model to recognize patterns"),
    ("what is a transformer", "a transformer is a model architecture that uses attention to understand context"),
    ("why are small models useful", "small models are fast, cheap, and easier to train on limited hardware"),
    ("what is a token", "a token is a unit of text such as a word, character, or subword"),
    ("explain learning", "learning means adjusting model weights to reduce prediction errors"),
    ("why do models learn", "models learn by repeatedly comparing predictions to real data and updating internal weights"),
    ("can you write code", "yes, i can continue code and technical text based on the prompt"),
    ("give me a joke", "why did the model cross the road? because it was trained on good data"),
    ("goodbye", "goodbye! keep building and experimenting with small ai models"),
    ("tiny llm", "a tiny llm is a compact language model built to be light and efficient"),
    ("small ai", "small ai is useful for demos, experimentation, education, and lightweight tools"),
    ("what is training", "training is the process of updating a model using examples so it gets better at prediction"),
    ("what is inference", "inference is using the trained model to generate new predictions from a prompt"),
    ("can you help me code", "yes, i can help with code snippets, ideas, and short technical explanations"),
    ("what is data", "data is the information used to teach the model patterns and relationships"),
    ("hello world", "hello world from a compact ai system built for learning and testing"),
    ("who made you", "i was made by a developer experimenting with a small transformer model"),
    ("what do you like", "i like small datasets, clean code, simple ideas, and efficient models"),
    ("what is deep learning", "deep learning is a type of machine learning based on neural networks"),
    ("what is a neural network", "a neural network is a model made of layers that transform input data into useful output"),
    ("what is a checkpoint", "a checkpoint is a saved snapshot of the model weights after training"),
    ("what is a dataset", "a dataset is a collection of examples used to teach the model patterns"),
    ("what is a transformer layer", "a transformer layer applies attention and feedforward operations to process tokens"),
    ("what is attention", "attention helps the model decide which parts of the text matter most"),
    ("what is a prompt", "a prompt is the text we provide so the model can generate a response"),
    ("how does a model learn", "a model learns by comparing predictions with the real target and adjusting its weights"),
    ("why is context important", "context matters because the next token depends on what came before"),
    ("why are small models useful", "small models are easier to train, faster to run, and useful for prototypes"),
    ("how is text represented", "text is represented as tokens, usually as integers mapped to characters or words"),
    ("what is overfitting", "overfitting happens when a model memorizes training text too closely and generalizes poorly"),
    ("what is generation", "generation is the process of producing new tokens one by one from a prompt"),
    ("what is a loss", "loss measures how wrong a model's predictions are during training"),
    ("what is a learning rate", "a learning rate controls how much the model updates its weights each step"),
    ("what is a batch", "a batch is a group of training examples processed together"),
    ("why do we use randomness", "randomness helps the model explore different possible continuations during generation"),
    ("what is a toy model", "a toy model is a small experimental model designed to learn basic ideas"),
    ("what is a checkpoint file", "a checkpoint file stores the trained model weights and settings"),
    ("what is deployment", "deployment is putting a trained model into use so others can interact with it"),
    ("what is coding", "coding is the act of writing logic and instructions so a program can do useful work"),
    ("what are examples", "examples are specific test cases used to teach a system what patterns to follow"),
]

GENERIC_TEXT = [
    "the quick brown fox jumps over the lazy dog\n",
    "small models can still learn useful patterns from data\n",
    "language models predict the next token based on prior context\n",
    "training on small data is a good way to prototype ideas quickly\n",
    "compact ai models are practical for demos and education\n",
    "a smart but small model can still generate helpful text\n",
    "github actions can automate model training after every push\n",
    "tiny models remain useful when speed and efficiency matter\n",
    "ai research often starts with a compact prototype before scaling up\n",
    "software engineering is about building working systems that solve real needs\n",
    "you can learn a lot by writing a tiny model and reading its outputs\n",
    "many useful ai systems are built from small, well-tested components\n",
    "language models often look like magic but they are really pattern predictors\n",
    "model training is iterative and feedback-driven\n",
    "a well-labeled dataset makes learning much easier\n",
    "developers use scripts to automate repetitive tasks and experiments\n",
    "the more structured the examples, the easier it is for a model to learn\n",
    "interesting ideas often begin with a minimal prototype\n",
    "open source projects thrive on clear documentation and simple examples\n",
    "next-token prediction is a core idea behind modern language models\n",
    "small-scale experiments reveal which ideas are worth scaling up\n",
    "a good model is useful only when it solves a problem reliably\n",
    "simple code and strong examples are often more valuable than complexity\n",
    "machine learning is basically pattern discovery through optimization\n",
    "models update their internal weights based on feedback from errors\n",
]

# ============================================================
# Model implementation
# ============================================================

@dataclass
class TrainingConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = BLOCK_SIZE
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.1
    learning_rate: float = 3e-3
    batch_size: int = 16
    max_iters: int = 3000
    eval_interval: int = 200
    generate_length: int = 80
    seed: int = 1337


class TinyGPT(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "ln_1": nn.LayerNorm(config.n_embd),
                "attn": nn.MultiheadAttention(config.n_embd, config.n_head, dropout=config.dropout, batch_first=True),
                "ln_2": nn.LayerNorm(config.n_embd),
                "mlp": nn.Sequential(
                    nn.Linear(config.n_embd, 4 * config.n_embd),
                    nn.GELU(),
                    nn.Linear(4 * config.n_embd, config.n_embd),
                    nn.Dropout(config.dropout),
                ),
            })
            for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok + pos.unsqueeze(0).expand(B, -1, -1)

        for block in self.blocks:
            residual = x
            x = block["ln_1"](x)
            attn_out, _ = block["attn"](x, x, x, need_weights=False)
            x = residual + attn_out
            residual = x
            x = block["ln_2"](x)
            x = residual + block["mlp"](x)

        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


class CharDataset(Dataset):
    def __init__(self, texts, block_size):
        self.texts = texts
        self.block_size = block_size
        self.tokens = []
        for text in texts:
            self.tokens.extend([stoi[ch] for ch in text])

    def __len__(self):
        return max(1, len(self.tokens) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def build_dataset(texts):
    dataset = CharDataset(texts, BLOCK_SIZE)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader


def generate_text(model, prompt, max_new_tokens=80, temperature=0.9):
    model.eval()
    input_ids = [stoi[ch] for ch in prompt]
    generated = input_ids[:]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([generated[-BLOCK_SIZE:]], dtype=torch.long)
            logits, _ = model(x)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            if len(generated) > BLOCK_SIZE + max_new_tokens:
                break
    return "".join(itos[i] for i in generated)


def train_model(config: TrainingConfig, texts):
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    model = TinyGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loader = build_dataset(texts)

    for step in range(config.max_iters):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits, loss = model(xb, yb)
            loss.backward()
            optimizer.step()
            break

        if step % config.eval_interval == 0:
            print(f"step {step}: loss {loss.item():.4f}")

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny GPT toy model")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max_iters", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--temperature", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainingConfig(seed=args.seed, max_iters=args.max_iters)
    texts = list(PROMPT_RESPONSES)
    texts.extend((prompt, text) for prompt, text in PROMPT_RESPONSES)
    texts = [prompt + "\n" + response for prompt, response in PROMPT_RESPONSES]
    texts += GENERIC_TEXT

    model = train_model(config, texts)

    if args.interactive:
        print("Tiny GPT ready. Type 'exit' to quit.")
        while True:
            prompt = input("You: ")
            if prompt.lower() in {"exit", "quit"}:
                break
            response = generate_text(model, prompt, max_new_tokens=80, temperature=args.temperature)
            print(f"Model: {response}")
    else:
        prompt = "hello"
        response = generate_text(model, prompt, max_new_tokens=80, temperature=args.temperature)
        print(response)


if __name__ == "__main__":
    main()
