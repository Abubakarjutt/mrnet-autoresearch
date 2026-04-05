import torch
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_vit import AdvancedMRNetViT, MultiScaleMRNetViT

# Test the models
print("Testing MRNet ViT models...")

# Test AdvancedMRNetViT
print("\n1. Testing AdvancedMRNetViT with vit_b_16:")
try:
    model = AdvancedMRNetViT(num_classes=3, model_name='vit_b_16', pretrained=True)
    print("Model created successfully!")

    # Create dummy input tensors (batch_size=1, slices=10, channels=3, height=224, width=224)
    batch_size = 1
    slices = 10
    dummy_input = torch.randn(batch_size, 3, 224, 224)  # ViT expects 3 channels, 224x224

    # Test forward pass
    sagittal = dummy_input
    coronal = dummy_input
    axial = dummy_input

    output = model(sagittal, coronal, axial)
    print(f"Output shape: {output.shape}")
    print("Forward pass successful!")

except Exception as e:
    print(f"Error in AdvancedMRNetViT: {e}")

# Test MultiScaleMRNetViT
print("\n2. Testing MultiScaleMRNetViT:")
try:
    model = MultiScaleMRNetViT(num_classes=3, pretrained=True)
    print("Multi-scale model created successfully!")

    # Create dummy input tensors
    batch_size = 1
    slices = 10
    dummy_input = torch.randn(batch_size, 3, 224, 224)  # ViT expects 3 channels, 224x224

    # Test forward pass
    sagittal = dummy_input
    coronal = dummy_input
    axial = dummy_input

    output = model(sagittal, coronal, axial)
    print(f"Output shape: {output.shape}")
    print("Forward pass successful!")

except Exception as e:
    print(f"Error in MultiScaleMRNetViT: {e}")

print("\nAll tests completed!")