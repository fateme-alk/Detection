# VAEGAN for CICIDS2017

Adapts "Skip the Benchmark: Generating System-Level Data using Generative ML" (GLSVLSI 2024) for network attack detection.

## 📋 Purpose
Generate synthetic minority attack samples (Infiltration, WebAttack, BruteForce, DDoS) to balance CICIDS2017 dataset.

## 🏗️ Architecture
- **MLPVAE**: MLP Variational Autoencoder (simpler)
- **DCGAN**: Deep Convolutional GAN (more complex)
- **Binary Transformation**: 32-bit fixed-point with 12-bit fractional
- **Metrics**: MMD, SSD, PRD, COSS for quality evaluation

## 📂 Structure
```
vaegan_impl/
├── config/vae_config.yaml    # Settings
├── models/                   # MLPVAE & DCGAN
├── train/vae_train.py       # Training script
├── utils/                   # Data, metrics, transforms
└── outputs/                 # Generated data & models
```

## 🚀 Quick Start

1. **Setup**:
```bash
cd /home/fateme/detection
# Ensure CICIDS2017 data is in ./data/CICIDS2017/
```

2. **Train**:
```bash
python -m vaegan_impl.train.vae_train
```

3. **Generate**:
Synthetic data saved to `vaegan_impl/outputs/synthetic/`

4. **Use**:
Merge synthetic data with real data for balanced training.

## 🔗 Connection to Phase 1
- Input: Minority classes from Phase 1 analysis
- Output: Synthetic samples for those classes
- Next: Train classifier on balanced dataset

## ⚙️ Key Settings (config/vae_config.yaml)
- Data: Features, minority classes, binary representation
- MLPVAE: 64 latent dim, 128→64 hidden, 150 epochs
- DCGAN: 100 latent dim, 4 conv layers, 150 epochs
- Output: 5000 samples per minority class

## ✅ Verification
Run test script: `python test_vaegan.py`

## 📊 Integration Workflow
```
Phase 1 (Analysis) → Identify minority classes
     ↓
Phase 2 (VAEGAN) → Generate synthetic samples
     ↓
Phase 3 (Training) → Balanced classifier
     ↓
Phase 4 (Evaluation) → Improved detection
```

**Status**: ✅ Implementation complete. Ready to generate synthetic data for minority network attacks.