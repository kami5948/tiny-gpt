# Tiny GPT

A compact GPT-style language model built to stay small, train quickly, and still behave much more like a real language model than a pure character predictor.

## What this project is
This project is a lightweight language model inspired by GPT-style architectures. It uses a very small transformer model with character-level tokenization and a small training dataset made of prompt-response pairs.

The goal is to produce a tiny but surprisingly capable model that is:
- easy to understand
- lightweight enough to run on modest hardware
- simple enough for educational or experimental use
- suitable for GitHub-hosted automation through GitHub Actions

## Why this version exists
Most beginner language model projects either:
- are extremely basic character predictors
- or are too large and complex to run easily

This version balances simplicity and structure:
- small transformer backbone
- compact vocabulary
- prompt-response training examples
- generation support
- interactive chat-like mode
- model checkpoint export
- GitHub Actions training workflow support

## Project goals
- learn the basics of next-token prediction
- experiment with small transformer-style architectures
- generate text from prompt inputs
- train quickly on CPU
- keep the repository compact and beginner-friendly
- make it possible to train automatically on GitHub

## Repository structure
- `train.py` — model definition, training code, generation logic
- `requirements.txt` — Python dependencies
- `.gitignore` — files to ignore
- `README.md` — project overview and instructions
- `LICENSE` — open-source license
- `.github/workflows/train.yml` — GitHub Actions workflow for automated training

## Install
```bash
pip install -r requirements.txt
