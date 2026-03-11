ML Experiment Tracker (Built From Scratch)

This project implements a small machine learning experiment framework from scratch using PyTorch.

The goal of the project is to learn how real ML research repositories are structured by building the infrastructure step-by-step.

The system supports:

- Reproducible experiments
- Automatic run tracking
- Config-driven training
- Model checkpointing
- Experiment evaluation
- Hyperparameter sweeps
- Leaderboard comparison

All experiments are stored in a structured `runs/` directory so results can be reproduced and compared easily.

---

Project Motivation

Most ML repositories hide the infrastructure behind frameworks like:

PyTorch Lightning

Hydra

MLflow

Weights & Biases

While those tools are powerful, they abstract away the engineering that makes ML experimentation reproducible.

This project builds a small experiment framework from scratch to understand:

how experiments are tracked

how training runs are logged

how models are reproduced

how hyperparameters are compared

The result is a clean and lightweight ML pipeline implemented entirely in Python.

# Project Structure

project_root/

configs/
    exp1.yaml              # experiment configuration

src/
    train.py               # main training script
    eval.py                # evaluation script for saved runs
    leaderboard.py         # ranks runs by final loss
    sweep.py               # runs multiple experiments automatically

    data.py                # dataset generation
    model.py               # model architecture
    trainer.py             # training loop

runs/
    <run_id>/              # automatically created experiment folders
        config.yaml
        metadata.json
        metrics.json
        model.pt


---

# Day-by-Day Development

Day 1 — Project Setup  
Created the initial project structure separating:

configs/
src/
runs/

This separates experiment configuration, code, and experiment outputs.

---

Day 2 — Configuration System  
Added YAML based configuration for experiments.

configs/exp1.yaml defines:
- dataset parameters
- training parameters
- random seed

This allows experiments to be controlled through config files instead of hardcoding values.

---

Day 3 — Experiment Tracking  
Implemented the experiment runner inside train.py.

Features added:
- unique run IDs
- automatic run folders
- metadata tracking
- metrics logging

Every training run creates:

runs/<run_id>/

---

Day 4 — Training Pipeline  
Added the actual PyTorch training loop.

Implemented:
- dataset generation
- linear regression model
- MSE loss
- SGD optimizer
- loss tracking across epochs

Training metrics are saved in metrics.json.

---

Day 5 — Evaluation Script  
Implemented eval.py.

This script:
- loads saved model checkpoint
- regenerates the dataset
- runs a forward pass
- recomputes loss
- verifies reproducibility of experiments

Command:

python src/eval.py --run runs/<run_id>

---

Day 6 — Leaderboard  
Implemented leaderboard.py.

This script scans all runs and ranks them based on final loss.

Command:

python src/leaderboard.py --top 10

This allows easy comparison between experiments.

---

Day 7 — CLI Overrides  
Added command line overrides for experiment parameters.

Example:

python src/train.py --lr 0.01 --epochs 20

Overrides values defined in the YAML configuration.

The resolved configuration is stored inside each run folder.

---

Day 8 — Hyperparameter Sweep  
Implemented sweep.py.

This script automatically launches multiple experiments using different hyperparameter combinations.

Example grid:

learning rates: [0.001, 0.01, 0.1]  
noise levels:   [0.05, 0.1, 0.2]

Each combination runs a new training experiment.

---

Day 9 — Modular Refactoring  
Refactored the project into modular components.

Moved logic into separate modules:

data.py
    dataset generation

model.py
    model architecture

trainer.py
    PyTorch training loop

train.py now acts as the experiment orchestrator.

---

# Result

The project now supports:

- reproducible experiments
- experiment tracking
- model checkpointing
- evaluation verification
- hyperparameter sweeps
- experiment ranking

All experiments are stored under the runs/ directory and can be reproduced at any time.
