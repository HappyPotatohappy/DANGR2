# DANGR: Domain-Adversarial Noise Generation with Regularization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **DANGR** (Domain-Adversarial Noise Generation with Regularization) for domain generalization.

## 🎯 Overview

DANGR is a novel approach for domain generalization that uses adversarial noise generation to create domain-invariant representations. The method consists of:

- **Noise Generator (G)**: Generates controlled perturbations to input images
- **Domain Classifier (D)**: Attempts to identify the domain of perturbed images
- **Class Classifier (C)**: Predicts the class labels
- **Feature Extractor (F)**: Extracts domain-invariant features

## 🚀 Quick Start

### Installation

```bash
# Clone repository with submodules
git clone --recursive https://github.com/HappyPotatohappy/DANGR2.git
cd DANGR2

# Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies

pip install -r requirements.txt

# Setup DomainBed
bash scripts/setup_domainbed.sh
```
# Training
## Standalone Version
```bash
python scripts/train_standalone.py \
    --data_dir ./data \
    --dataset PACS \
    --test_env 0 \
    --epochs 100
```
## DomainBed Integration
python -m domainbed.scripts.train \
    --data_dir ./data \
    --algorithm DANGR \
    --dataset PACS \
    --test_env 0
```
# 📊 Results

