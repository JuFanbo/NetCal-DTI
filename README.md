# NetCal-DTI: Network Topology-Calibrated Drug-Target Interaction Prediction

## Overview

NetCal-DTI is a hybrid framework that reframes DTI prediction as a confidence calibration problem under biological structural constraints. By leveraging Laplacian eigenvectors (LEs) to encode the global topology of the DTI interactome, NetCal-DTI introduces a Network Calibration Module that acts as a structural adjudicator, dynamically recalibrating inductive predictions against network-level consistency. Evaluations across three benchmarks (DrugBank, BIOSNAP and BindingDB) demonstrate that NetCal-DTI achieves substantial precision gains while maintaining competitive recall and AUROC. Crucially, case studies reveal that network calibration suppresses 59.2% of false positives at the cost of only a minimal increase in new errors. Molecular docking validates the superior binding energetics of high-confidence candidates, and pathway enrichment analyses confirm the biological coherence of predictions for the HDAC inhibitor Tucidinostat. NetCal-DTI thus establishes a new paradigm for trustworthy DTI prediction, where global network topology serves not merely as an additional feature but as a principled regularizer for confidence calibration. 

## Key Features

- **Network Calibration**: Dynamically recalibrates inductive predictions using network-level consistency
- **Hybrid Architecture**: Combines inductive molecular encoding with transductive topological constraints
- **High Precision**: Significantly reduces false positives while maintaining competitive recall
- **Biological Plausibility**: Ensures predictions align with established biological regularities
- **Practical Utility**: Designed for real-world drug discovery applications

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (recommended for GPU acceleration)

### Quick Installation
```bash
git clone https://github.com/JuFanbo/NetCal-DTI.git
cd NetCal-DTI
pip install -r requirements.txt
```

## Quick Start

### Basic Usage
```python
from main import perf, ablation

# Run comparative performance evaluation
perf()

# Run ablation studies
ablation()
```

### Custom Training
```python
from utils import performance
from NetCalDTI import NetCalDTI

# Train NetCal-DTI model
performance(
    "NetCal-DTI",
    NetCalDTI,
    batch_size=256,
    lr=5e-4,
    runs=5
)
```

## Repository Structure

```
NetCal-DTI/
├── main.py                 # Main execution script
├── NetCalDTI.py           # Core model implementation
├── dataloader.py          # Data loading utilities
├── utils.py               # Utility functions
├── case.py                # Case study implementations
├── requirements.txt       # Python dependencies
│
├── Benchmark/             # Comparative models
│   ├── attention.py       # Attention-based model
│   ├── CNNGIN.py         # CNN+GIN baseline
│   ├── GraphDTA.py       # GraphDTA implementation
│   ├── DrugBAN.py        # DrugBAN implementation
│   ├── ML_DTI.py         # ML_DTI implementation
│   ├── HyperAttentionDTI.py
│   ├── TransformerCPI.py
│   └── le.py             # Laplacian eigenvectors only
│
├── Data/                  # Datasets and embeddings
│   ├── drugbank.csv       # DrugBank dataset
│   ├── bindingdb.csv      # BindingDB dataset
│   ├── biosnap.csv        # BIOSNAP dataset
│   ├── drug_info.csv      # Drug metadata
│   ├── seq.csv           # Protein sequences
│   ├── smiles.csv        # SMILES strings
│   ├── *.pickle          # Preprocessed graph objects
│   └── esm.pickle        # ESM protein embeddings
│
└── docking/              # Molecular docking validation
    ├── dock.py           # Docking scripts
    ├── CDK2/            # CDK2 target data
    └── SERT/            # SERT target data
```

## Model Architecture

NetCal-DTI consists of three core components:

### 1. Inductive Evidence Encoding
- Molecular graph encoding using Graph Isomorphism Networks (GIN)
- Protein sequence processing with ResNet-style 1D-CNN and ESM embeddings
- Cross-attention mechanism for fine-grained interaction modeling

### 2. Network Calibration Module
- Constructs heterogeneous DTI network from training interactions
- Encodes global topology using Laplacian Eigenvectors (LEs)
- Generates calibration vectors representing structural confidence

### 3. Calibrated Consensus Fusion
- Concatenates inductive and calibration vectors
- Processes through multilayer perceptron (MLP)
- Produces final calibrated interaction scores

## Available Datasets

The framework supports multiple standard DTI benchmarks:
- **DrugBank**: 46,192 DTIs involving 7,968 drugs and 4,359 targets
- **BIOSNAP**: 26,558 balanced DTIs across 4,421 drugs and 1,980 targets
- **BindingDB**: 48,548 affinity records with Kd values

## Benchmark Models

Implemented state-of-the-art models for comparison:
- **GraphDTA**: Graph neural network with GIN architecture
- **HyperAttentionDTI**: End-to-end model with hyper-attention mechanism
- **ML_DTI**: Hybrid approach combining learned and handcrafted features
- **TransformerCPI**: Transformer-based sequence-to-sequence model
- **DrugBAN**: Bilinear attention network with domain adaptation


## Key Methodological Contributions

1. **Systematic Overconfidence Mitigation**: Identifies and addresses systematic overconfidence in inductive DTI prediction
2. **Network-Aware Calibration**: Leverages global DTI topology as structural prior
3. **Multi-Scale Validation**: Comprehensive evaluation spanning benchmarks, molecular docking, and pathway analysis


## Contact

- **Maintainers**: Fanbo Ju, Yuxin Zhang, Longfei Luo
- **Corresponding Authors**: 
  - Bin Yang (yangbin1@suda.edu.cn)
  - Guang Hu (huguang@suda.edu.cn)
- **Institution**: School of Life Sciences, Suzhou Medical College of Soochow University
- **Repository**: https://github.com/JuFanbo/NetCal-DTI


## Acknowledgments

This work was supported by:
- National Natural Science Foundation of China (32271292)
- Natural Science Foundation of Jiangsu Province (BK20255001)
- MOE Key Laboratory of Geriatric Diseases and Immunology (JYN202404)
- Priority Academic Program Development (PAPD) of Jiangsu Higher Education Institutions

---

*Note: This project is under active development. Please check for updates regularly.*
