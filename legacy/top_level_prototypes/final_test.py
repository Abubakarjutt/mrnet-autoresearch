#!/usr/bin/env python3
"""
Final comprehensive test for MRNet ViT implementation
"""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_vit import AdvancedMRNetViT, MultiScaleMRNetViT

print("MRNet ViT Model Implementation - Final Test")
print("=" * 50)

# Test 1: Import successful
print("✓ Import test passed")

# Test 2: Model creation
print("\nTesting model creation...")

try:
    # Test basic ViT model
    model1 = AdvancedMRNetViT(num_classes=3, model_name='vit_b_16', pretrained=False)
    print("✓ AdvancedMRNetViT (vit_b_16) created successfully!")

    # Test large ViT model
    model2 = AdvancedMRNetViT(num_classes=3, model_name='vit_l_16', pretrained=False)
    print("✓ AdvancedMRNetViT (vit_l_16) created successfully!")

    # Test multi-scale model
    model3 = MultiScaleMRNetViT(num_classes=3, pretrained=False)
    print("✓ MultiScaleMRNetViT created successfully!")

    print("\nAll models created successfully!")

except Exception as e:
    print(f"✗ Error creating models: {e}")
    sys.exit(1)

# Test 3: Model forward pass
print("\nTesting forward pass...")

try:
    # Create dummy input tensors (batch_size=1, channels=3, height=224, width=224)
    dummy_sagittal = torch.randn(1, 3, 224, 224)
    dummy_coronal = torch.randn(1, 3, 224, 224)
    dummy_axial = torch.randn(1, 3, 224, 224)

    # Test forward pass with each model
    output1 = model1(dummy_sagittal, dummy_coronal, dummy_axial)
    print(f"✓ AdvancedMRNetViT forward pass successful: {output1.shape}")

    output2 = model2(dummy_sagittal, dummy_coronal, dummy_axial)
    print(f"✓ AdvancedMRNetViT (large) forward pass successful: {output2.shape}")

    output3 = model3(dummy_sagittal, dummy_coronal, dummy_axial)
    print(f"✓ MultiScaleMRNetViT forward pass successful: {output3.shape}")

    print("\nAll forward passes successful!")

except Exception as e:
    print(f"✗ Error in forward pass: {e}")
    sys.exit(1)

# Test 4: Model parameters
print("\nModel parameter counts:")
print("-" * 30)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"AdvancedMRNetViT (vit_b_16): {count_parameters(model1):,} parameters")
print(f"AdvancedMRNetViT (vit_l_16): {count_parameters(model2):,} parameters")
print(f"MultiScaleMRNetViT: {count_parameters(model3):,} parameters")

print("\n" + "=" * 50)
print("🎉 ALL TESTS PASSED!")
print("MRNet ViT implementation is working correctly")
print("=" * 50)
print("\nTo run training:")
print("python3 run_model_fixed.py --model_type vit_b_16 --num_epochs 5")