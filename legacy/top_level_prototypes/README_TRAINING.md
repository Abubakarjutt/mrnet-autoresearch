# MRNet ViT Training Script

This directory contains a complete training implementation for MRNet knee injury diagnosis using Vision Transformers.

## Files

1. `train_vit.py` - Main training script with configurable model options
2. `test_model.py` - Model testing script
3. `advanced_vit.py` - Vision Transformer implementations
4. `requirements.txt` - Required Python packages
5. `README.md` - Project documentation

## How to Run Training

```bash
# Install dependencies
pip install -r requirements.txt

# Run training (default settings)
python train_vit.py

# Run training with custom settings
python train_vit.py --model_type vit_b_16 --num_epochs 20 --batch_size 4 --learning_rate 1e-4
```

## Model Options

- `vit_b_16` - Base ViT with 16x16 patches
- `vit_l_16` - Large ViT with 16x16 patches
- `vit_h_14` - Huge ViT with 14x14 patches
- `multiscale` - Multi-scale ViT approach

## Training Data Structure

The training data should be organized as:
```
MRNet-v1.0/train/
├── axial/
├── coronal/
└── sagittal/
```

Each directory should contain `.npy` files with patient IDs.

## Requirements

- Python 3.7+
- PyTorch 1.10+
- torchvision
- pandas
- numpy
- scikit-learn

The training script will automatically:
1. Load MRNet data
2. Split into train/validation sets
3. Train the specified ViT model
4. Save best performing model
5. Output training metrics

Note: Training on a GPU is highly recommended for performance.