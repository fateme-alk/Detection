#!/usr/bin/env python3
"""
Test script for VAEGAN Implementation

This script tests the basic functionality of the VAEGAN framework
for CICIDS2017 network traffic detection.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd

# Add VAEGAN to path
sys.path.insert(0, os.path.abspath('.'))

# Test imports
print("=== Testing VAEGAN Framework ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

try:
    from vaegan_impl.utils.metrics import mmd_rbf, SSD, PRD, COSS
    print("✓ Metrics module imported successfully")
except ImportError as e:
    print(f"✗ Error importing metrics: {e}")

try:
    from vaegan_impl.utils.data_transform import BinaryDataTransformer
    print("✓ DataTransform module imported successfully")
except ImportError as e:
    print(f"✗ Error importing data_transform: {e}")

try:
    from vaegan_impl.utils.dataset import NetworkTrafficDataset
    print("✓ Dataset module imported successfully")
except ImportError as e:
    print(f"✗ Error importing dataset: {e}")

try:
    from vaegan_impl.models.mlp_vae import LinearVAE
    print("✓ MLPVAE module imported successfully")
except ImportError as e:
    print(f"✗ Error importing mlp_vae: {e}")

try:
    from vaegan_impl.models.dcn_vae import DCGAN
    print("✓ DCGAN module imported successfully")
except ImportError as e:
    print(f"✗ Error importing dcn_vae: {e}")

print("\n=== Running Tests ===")

# Test 1: Metrics
print("1. Testing metrics functions...")
try:
    # Create dummy data
    real_data = torch.randn(32, 64)
    fake_data = torch.randn(32, 64)
    
    mmd = mmd_rbf(real_data, fake_data)
    ssd = SSD(real_data, fake_data).mean()
    prd = PRD(real_data, fake_data).mean()
    coss = COSS(real_data, fake_data).mean()
    
    print(f"   MMD: {mmd:.4f}")
    print(f"   SSD: {ssd:.4f}")
    print(f"   PRD: {prd:.4f}")
    print(f"   COSS: {coss:.4f}")
    print("   ✓ Metrics computed successfully")
except Exception as e:
    print(f"   ✗ Error computing metrics: {e}")

# Test 2: Data transformer
print("2. Testing data transformer...")
try:
    transformer = BinaryDataTransformer(bit_length=32, frac_bits=12)
    print("   ✓ BinaryDataTransformer created")
    
    # Test transformation on dummy data
    dummy_df = pd.DataFrame({
        'feature1': [0.5, 0.8, 0.3],
        'feature2': [0.7, 0.2, 0.9],
        'feature3': [0.1, 0.6, 0.4]
    })
    
    # Need to fit first
    transformer.fit(dummy_df)
    
    # Transform
    binary_data = transformer.transform(dummy_df)
    print(f"   ✓ Data transformed to shape: {binary_data.shape}")
except Exception as e:
    print(f"   ✗ Error with transformer: {e}")

# Test 3: Model creation
print("3. Testing model creation...")
try:
    # Test MLPVAE
    mlp_model = LinearVAE(input_dim=128, latent_dim=64, hidden_dims=[128, 64])
    print(f"   ✓ MLPVAE created with {sum(p.numel() for p in mlp_model.parameters())} parameters")
    
    # Test DCGAN
    dcgan_model = DCGAN(latent_dim=100, num_features=64)
    print(f"   ✓ DCGAN created with {sum(p.numel() for p in dcgan_model.parameters())} parameters")
    
    print("   ✓ Both models created successfully")
except Exception as e:
    print(f"   ✗ Error creating models: {e}")

# Test 4: Configuration
print("4. Testing configuration...")
try:
    import yaml
    
    config_path = './vaegan_impl/config/vae_config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print(f"   ✓ Configuration loaded: {len(config)} sections")
            print(f"   - Data features: {len(config.get('data', {}).get('features', []))}")
            print(f"   - MLPVAE settings: {len(config.get('mlp_vae', {}))}")
            print(f"   - DCGAN settings: {len(config.get('dcgan', {}))}")
    else:
        print(f"   ✗ Configuration file not found: {config_path}")
except Exception as e:
    print(f"   ✗ Error with configuration: {e}")

print("\n=== Environment Check ===")
# Check CUDA
if torch.cuda.is_available():
    print(f"✓ CUDA is available ({torch.cuda.get_device_name(0)})")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("✗ CUDA is not available (will run on CPU)")

# Check data directory
data_dir = './data/CICIDS2017/'
if os.path.exists(data_dir):
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"✓ Data directory exists with {len(csv_files)} CSV files")
else:
    print(f"✗ Data directory not found: {data_dir}")
    print("  Note: You'll need to download CICIDS2017 dataset")

print("\n=== Installation Summary ===")
print("If you see no major errors above, the VAEGAN framework is installed correctly.")
print("Next steps:")
print("1. Make sure you have CICIDS2017 data in ./data/CICIDS2017/")
print("2. Run the training script: python vaegan_impl/train/vae_train.py")
print("3. Check generated synthetic data in vaegan_impl/outputs/synthetic/")
print("\nFor debugging:")
print("- Check that all dependencies are installed (torch, pandas, numpy, yaml)")
print("- Verify your data files are in the correct location")
print("- Adjust configuration in vaegan_impl/config/vae_config.yaml")

print("\nDone! ✓")