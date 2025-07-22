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
git clone --recursive https://github.com/YOUR_USERNAME/DANGR-DomainBed.git
cd DANGR-DomainBed

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup DomainBed
bash scripts/setup_domainbed.sh
```

### Training

#### Standalone Version
```bash
python scripts/train_standalone.py \
    --data_dir ./data \
    --dataset PACS \
    --test_env 0 \
    --epochs 100
```

#### DomainBed Integration
```bash
python -m domainbed.scripts.train \
    --data_dir ./data \
    --algorithm DANGR \
    --dataset PACS \
    --test_env 0
```

## 📊 Results

| Dataset | Art | Cartoon | Photo | Sketch | Average |
|---------|-----|---------|-------|--------|---------|
| PACS    | 85.2| 78.9    | 96.1  | 79.5   | 84.9±7.3|

## 📁 Project Structure

```
DANGR-DomainBed/
├── dangr/          # Core DANGR implementation
├── scripts/        # Training and evaluation scripts
├── experiments/    # Experiment configurations
├── docs/           # Documentation
└── domainbed/      # DomainBed submodule
```

## 📖 Documentation

- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api.md)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- This work builds upon [DomainBed](https://github.com/facebookresearch/DomainBed)
- Thanks to the authors of the original DomainBed paper

## 📧 Contact

- Your Name - your.email@example.com
- Project Link: https://github.com/YOUR_USERNAME/DANGR-DomainBed
