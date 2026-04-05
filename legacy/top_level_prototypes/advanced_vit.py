import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights, ViT_L_16_Weights, ViT_H_14_Weights

class AdvancedMRNetViT(nn.Module):
    def __init__(self, num_classes=3, model_name='vit_b_16', pretrained=True):
        super().__init__()

        # Select and initialize the appropriate ViT model
        if model_name == 'vit_b_16':
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit_model = models.vit_b_16(weights=weights)
            self.embedding_dim = 768
        elif model_name == 'vit_l_16':
            weights = ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit_model = models.vit_l_16(weights=weights)
            self.embedding_dim = 1024
        elif model_name == 'vit_h_14':
            weights = ViT_H_14_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit_model = models.vit_h_14(weights=weights)
            self.embedding_dim = 1280
        else:
            raise ValueError("Unsupported model name. Choose from: vit_b_16, vit_l_16, vit_h_14")

        # Replace the classifier head to output our number of classes
        # For ViT models, we need to modify the head properly
        if hasattr(self.vit_model, 'heads'):
            # ViT models have 'heads' attribute for classification
            self.vit_model.heads = nn.Linear(self.embedding_dim, num_classes)
        elif hasattr(self.vit_model, 'classifier'):
            # Some ViT versions have 'classifier'
            self.vit_model.classifier = nn.Linear(self.embedding_dim, num_classes)
        else:
            # If no specific classifier attribute, create a new one
            self.vit_model.heads = nn.Linear(self.embedding_dim, num_classes)

        # Keep the original model for feature extraction
        self.vit_model_features = self.vit_model

    def forward(self, sagittal, coronal, axial):
        # Process each view through the ViT model
        # For ViT models, the forward pass returns logits directly
        # We'll extract features and then classify

        # Get features from each view
        features_sagittal = self.vit_model_features(sagittal)
        features_coronal = self.vit_model_features(coronal)
        features_axial = self.vit_model_features(axial)

        # For simplicity, we'll use the first view's features and average
        # In a real implementation, you'd want to properly fuse features
        combined_features = (features_sagittal + features_coronal + features_axial) / 3

        return combined_features

class MultiScaleMRNetViT(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()

        # Initialize multiple ViT models with different scales
        self.vit_b_16 = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.vit_l_16 = models.vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None)

        # Replace classifier heads to output features instead of classes
        for model in [self.vit_b_16, self.vit_l_16]:
            if hasattr(model, 'heads'):
                model.heads = nn.Identity()
            elif hasattr(model, 'classifier'):
                model.classifier = nn.Identity()

        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(768 + 1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, sagittal, coronal, axial):
        # Process each view through the ViT models separately
        features_b16_sagittal = self.vit_b_16(sagittal)
        features_b16_coronal = self.vit_b_16(coronal)
        features_b16_axial = self.vit_b_16(axial)

        features_l16_sagittal = self.vit_l_16(sagittal)
        features_l16_coronal = self.vit_l_16(coronal)
        features_l16_axial = self.vit_l_16(axial)

        # Average features from each view
        avg_b16 = (features_b16_sagittal + features_b16_coronal + features_b16_axial) / 3
        avg_l16 = (features_l16_sagittal + features_l16_coronal + features_l16_axial) / 3

        # Combine features from different scales
        combined_features = torch.cat((avg_b16, avg_l16), dim=1)

        # Final classification
        output = self.feature_fusion(combined_features)
        return output