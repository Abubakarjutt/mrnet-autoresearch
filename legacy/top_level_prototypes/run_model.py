#!/usr/bin/env python3

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

# Import our model
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from advanced_vit import AdvancedMRNetViT, MultiScaleMRNetViT

# Custom Dataset class for MRNet
class MRNetDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the patient ID
        patient_id = self.data.iloc[idx]['id']

        # Load the three views
        sagittal_path = os.path.join(self.data_dir, 'sagittal', f'{patient_id:04d}.npy')
        coronal_path = os.path.join(self.data_dir, 'coronal', f'{patient_id:04d}.npy')
        axial_path = os.path.join(self.data_dir, 'axial', f'{patient_id:04d}.npy')

        # Load numpy arrays
        sagittal = np.load(sagittal_path)
        coronal = np.load(coronal_path)
        axial = np.load(axial_path)

        # Convert to tensors and normalize
        sagittal = torch.from_numpy(sagittal).float()
        coronal = torch.from_numpy(coronal).float()
        axial = torch.from_numpy(axial).float()

        # Add channel dimension if needed (assuming data is already in correct format)
        if sagittal.dim() == 3:
            sagittal = sagittal.unsqueeze(0)  # Add batch dimension
        if coronal.dim() == 3:
            coronal = coronal.unsqueeze(0)
        if axial.dim() == 3:
            axial = axial.unsqueeze(0)

        # Get label
        label = self.data.iloc[idx]['label']

        return sagittal, coronal, axial, label

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4, device='cuda'):
    # Move model to device
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []

    print(f"Starting training on {device}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (sagittal, coronal, axial, labels) in enumerate(train_loader):
            sagittal, coronal, axial = sagittal.to(device), coronal.to(device), axial.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(sagittal, coronal, axial)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for sagittal, coronal, axial, labels in val_loader:
                sagittal, coronal, axial = sagittal.to(device), coronal.to(device), axial.to(device)
                labels = labels.to(device)

                outputs = model(sagittal, coronal, axial)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train
        val_acc = 100. * correct_val / total_val

        train_losses.append(train_loss)
        val_accuracies.append(val_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')
        print('-' * 50)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')

    print(f'Best validation accuracy: {best_val_acc:.2f}%')
    return train_losses, val_accuracies

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train MRNet ViT model')
    parser.add_argument('--model_type', type=str, default='vit_b_16',
                       choices=['vit_b_16', 'vit_l_16', 'vit_h_14', 'multiscale'],
                       help='Choose model type (default: vit_b_16)')
    parser.add_argument('--num_classes', type=int, default=3,
                       help='Number of classes (default: 3)')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--data_dir', type=str, default='MRNet-AI-assited-diagnosis-of-knee-injuries/MRNet-v1.0/train',
                       help='Path to training data directory')
    parser.add_argument('--csv_file', type=str, default='MRNet-AI-assited-diagnosis-of-knee-injuries/MRNet-v1.0/train-abnormal.csv',
                       help='Path to CSV file with training labels')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    if args.model_type == 'multiscale':
        model = MultiScaleMRNetViT(num_classes=args.num_classes, pretrained=True)
    else:
        model = AdvancedMRNetViT(num_classes=args.num_classes, model_name=args.model_type, pretrained=True)

    print(f"Model created: {args.model_type}")

    # Create datasets
    train_dataset = MRNetDataset(args.data_dir, args.csv_file)

    # Split data into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    print(f"Training set size: {len(train_subset)}")
    print(f"Validation set size: {len(val_subset)}")

    # Train model
    print("Starting training...")
    start_time = time.time()

    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device
    )

    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes")

    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    print("Final model saved as 'final_model.pth'")

if __name__ == "__main__":
    main()