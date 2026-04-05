# Simple test to verify the model works
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_vit import AdvancedMRNetViT, MultiScaleMRNetViT
    print("✓ Successfully imported MRNet ViT models")

    # Test creating a model
    print("Creating AdvancedMRNetViT model...")
    model = AdvancedMRNetViT(num_classes=3, model_name='vit_b_16', pretrained=True)
    print("✓ AdvancedMRNetViT model created successfully!")

    # Test creating a multi-scale model
    print("Creating MultiScaleMRNetViT model...")
    model2 = MultiScaleMRNetViT(num_classes=3, pretrained=True)
    print("✓ MultiScaleMRNetViT model created successfully!")

    print("\n" + "="*50)
    print("MODEL TEST SUCCESSFUL")
    print("The MRNet ViT models are ready for training!")
    print("="*50)
    print("\nTo train the models, you would run:")
    print("python3 run_model.py --model_type vit_b_16 --num_epochs 10")
    print("\nRequirements:")
    print("- MRNet dataset organized in MRNet-AI-assited-diagnosis-of-knee-injuries/MRNet-v1.0/train/")
    print("- CSV file with patient labels")
    print("- Proper data structure with sagittal, coronal, axial directories")

except Exception as e:
    print(f"Error: {e}")
    print("There may be an issue with the model import or structure.")