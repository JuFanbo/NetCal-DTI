# NetCal-DTI: Network Topology-Calibrated Drug-Target Interaction Prediction

## Overview

NetCal-DTI is a novel hybrid framework that addresses systematic overconfidence in inductive drug-target interaction (DTI) prediction by reframing it as a confidence calibration problem under biological structural constraints. The framework integrates inductive molecular evidence with global network topology calibration to produce more reliable and biologically plausible predictions.

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

## Molecular Docking Validation

The framework includes comprehensive molecular docking validation:

```bash
# Run docking validation for CDK2 target
python docking/dock.py --target CDK2 --config docking/CDK2/config.txt
```

## Key Methodological Contributions

1. **Systematic Overconfidence Mitigation**: Identifies and addresses systematic overconfidence in inductive DTI prediction
2. **Network-Aware Calibration**: Leverages global DTI topology as structural prior
3. **Multi-Scale Validation**: Comprehensive evaluation spanning benchmarks, molecular docking, and pathway analysis

## Applications

- **Virtual Screening**: High-precision candidate prioritization
- **Drug Repositioning**: Biologically coherent target prediction
- **Systems Pharmacology**: Multi-target interaction profiling

## Citation

If you use NetCal-DTI in your research, please cite our paper:

```bibtex
@article{ju2025netcal,
  title={NetCal-DTI: A Network Topology-Calibrated Hybrid Framework for High-Precision Inductive Drug-Target Interaction Prediction},
  author={Ju, Fanbo and Zhang, Yuxin and Luo, Longfei and Zhao, Qixuan and Yang, Bin and Hu, Guang},
  journal={Nature Communications},
  year={2025},
  doi={10.1038/s41467-025-XXXXX-X}
}
```

## Contact

- **Maintainers**: Fanbo Ju, Yuxin Zhang, Longfei Luo
- **Corresponding Authors**: 
  - Bin Yang (yangbin1@suda.edu.cn)
  - Guang Hu (huguang@suda.edu.cn)
- **Institution**: School of Life Sciences, Suzhou Medical College of Soochow University
- **Repository**: https://github.com/JuFanbo/NetCal-DTI

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This work was supported by:
- National Natural Science Foundation of China (32271292)
- Natural Science Foundation of Jiangsu Province (BK20255001)
- MOE Key Laboratory of Geriatric Diseases and Immunology (JYN202404)
- Priority Academic Program Development (PAPD) of Jiangsu Higher Education Institutions

---

*Note: This project is under active development. Please check for updates regularly.*
