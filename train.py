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
    ("can you explain yourself", "i explain by predicting likely text continuations from patterns in training data"),
    ("can you do math", "i can follow simple symbolic patterns, but i am not a formal calculator"),
    ("what is language modeling", "language modeling means predicting the next token in a sequence of text"),
    ("what is a vector", "a vector is a list of numbers representing information in a model"),
    ("what is a neural network", "a neural network is a structure of connected layers that transforms data"),
    ("what is a hidden layer", "a hidden layer sits between the input and output and captures internal patterns"),
    ("what is backpropagation", "backpropagation is the process of calculating how errors should change the model weights"),
    ("what is a tokenizer", "a tokenizer converts text into tokens that a model can process"),
    ("why do models sometimes repeat", "models may repeat because they predict the most likely next token several times in a row"),
    ("what is a demo model", "a demo model is a compact model meant to show a concept in a simple way"),
    ("what is brainstorming", "brainstorming is generating many ideas before choosing the best one"),
    ("what is a prototype", "a prototype is an early version of a system used to test a concept"),
    ("what is an experiment", "an experiment is a structured test to learn more about how a model behaves"),
    ("why test small models", "small models are easier to debug and give fast feedback during experimentation"),
    ("what is a sequence", "a sequence is an ordered list of tokens such as a sentence or paragraph"),
    ("what is randomness in generation", "randomness makes output varied instead of always choosing the same token"),
    ("what is a context window", "a context window is the number of previous tokens the model can look at"),
    ("what is coding ai", "coding ai is the use of language models to assist with programming and technical tasks"),
    ("what is a tiny model", "a tiny model is a compact model designed to be fast, small, and trainable on modest hardware"),
    ("what is text generation", "text generation is the process of creating new text from a prompt and learned patterns"),
    ("what is the best way to learn", "the best way to learn is to build small projects, test ideas, and iterate quickly"),
    ("what is a github repo", "a github repo is a project directory stored and versioned on github"),
    ("what is a workflow", "a workflow is an automated process that runs code steps for you"),
    ("what is automation", "automation is the process of letting software do repetitive tasks without manual steps"),
    ("what is a model checkpoint", "a model checkpoint is a saved copy of the model at a given training stage"),
    ("what is code", "code is a set of instructions written in a programming language"),
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
    "many](#)

