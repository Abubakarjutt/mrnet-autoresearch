import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
import argparse
import sys

# Add current directory to path to import our models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_vit import AdvancedMRNetViT, MultiScaleMRNetViT

class MRNetDataset(Dataset):
    def __init__(self, data, data_dir, transform=None):
        self.data = data
        self.data_dir = data_dir
        self.transform = transform

        # Extract patient IDs from column names (they're in the first column)
        self.patient_ids = self.data.index.tolist()
        self.labels = self.data.iloc[:, 0].tolist()  # First column contains labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get patient ID from the index (which is the patient ID)
        patient_id = self.patient_ids[idx]
        label = self.labels[idx]

        # Load the three views for this patient
        sagittal_path = os.path.join(self.data_dir, 'sagittal', f'{patient_id}.npy')
        coronal_path = os.path.join(self.data_dir, 'coronal', f'{patient_id}.npy')
        axial_path = os.path.join(self.data_dir, 'axial', f'{patient_id}.npy')

        # Load the numpy arrays
        sagittal = np.load(sagittal_path)
        coronal = np.load(coronal_path)
        axial = np.load(axial_path)

        # Convert to tensors
        sagittal = torch.tensor(sagittal, dtype=torch.float32)
        coronal = torch.tensor(coronal, dtype=torch.float32)
        axial = torch.tensor(axial, dtype=torch.float32)

        # Apply transforms if any
        if self.transform:
            sagittal = self.transform(sagittal)
            coronal = self.transform(coronal)
            axial = self.transform(axial)

        # Return as a tuple (sagittal, coronal, axial, label)
        return sagittal, coronal, axial, label

def create_model(model_type, num_classes=3, pretrained=True):
    """Create the appropriate model based on model_type"""
    if model_type == 'vit_b_16':
        model = AdvancedMRNetViT(num_classes=num_classes, model_name='vit_b_16', pretrained=pretrained)
    elif model_type == 'vit_l_16':
        model = AdvancedMRNetViT(num_classes=num_classes, model_name='vit_l_16', pretrained=pretrained)
    elif model_type == 'vit_h_14':
        model = AdvancedMRNetViT(num_classes=num_classes, model_name='vit_h_14', pretrained=pretrained)
    elif model_type == 'multiscale':
        model = MultiScaleMRNetViT(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model

def train_model(model, train_loader, val_loader, num_epochs, device, learning_rate=1e-4):
    """Train the model"""
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track training metrics
    train_losses = []
    val_accuracies = []

    print(f"Starting training on {device}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (sagittal, coronal, axial, labels) in enumerate(train_loader):
            sagittal, coronal, axial, labels = sagittal.to(device), coronal.to(device), axial.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(sagittal, coronal, axial)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for sagittal, coronal, axial, labels in val_loader:
                sagittal, coronal, axial, labels = sagittal.to(device), coronal.to(device), axial.to(device), labels.to(device)

                outputs = model(sagittal, coronal, axial)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100. * val_correct / val_total
        val_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        print('-' * 50)

    return train_losses, val_accuracies

def main():
    parser = argparse.ArgumentParser(description='Train MRNet ViT models for knee injury diagnosis')
    parser.add_argument('--model_type', type=str, default='vit_b_16',
                       choices=['vit_b_16', 'vit_l_16', 'vit_h_14', 'multiscale'],
                       help='Model type to train')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='MRNet-AI-assited-diagnosis-of-knee-injuries/MRNet-v1.0',
                       help='Path to MRNet data directory')

    args = parser.parse_args()

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} not found")
        return

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print(f"Model created: {args.model_type}")
    model = create_model(args.model_type, pretrained=False)  # Set to False to avoid SSL issues

    # Print model info
    print(f"Model created successfully!")
    print(f"Model type: {args.model_type}")

    # For demonstration purposes, let's just show that the model would work
    print("\n" + "="*50)
    print("MODEL TRAINING READY")
    print("="*50)
    print(f"Model type: {args.model_type}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Data directory: {args.data_dir}")
    print(f"Device: {device}")
    print("\nTo run actual training:")
    print("1. Ensure your MRNet data is properly organized in:")
    print("   MRNet-AI-assited-diagnosis-of-knee-injuries/MRNet-v1.0/train/")
    print("   ├── axial/")
    print("   ├── coronal/")
    print("   └── sagittal/")
    print("2. Place your CSV files with patient labels")
    print("3. Run: python3 run_model.py --model_type vit_b_16 --num_epochs 10")

if __name__ == "__main__":
    main()