#!/bin/bash

# Training script for different model configurations

echo "Training Advanced ViT-B/16 model..."
python train.py -t acl -p Segittal_Coronal_and_Axial --prefix_name acl_vit_b16 --epochs 50 --lr 1e-5 --model_type advanced --vit_model vit_b_16 --pretrained 1

echo "Training Advanced ViT-L/16 model..."
python train.py -t acl -p Segittal_Coronal_and_Axial --prefix_name acl_vit_l16 --epochs 50 --lr 1e-5 --model_type advanced --vit_model vit_l_16 --pretrained 1

echo "Training Multi-Scale model..."
python train.py -t acl -p Segittal_Coronal_and_Axial --prefix_name acl_multiscale --epochs 50 --lr 1e-5 --model_type multiscale --pretrained 1

echo "Training with different tasks..."
python train.py -t abnormal -p Segittal_Coronal_and_Axial --prefix_name abnormal_vit_b16 --epochs 50 --lr 1e-5 --model_type advanced --vit_model vit_b_16 --pretrained 1
python train.py -t meniscus -p Segittal_Coronal_and_Axial --prefix_name meniscus_vit_b16 --epochs 50 --lr 1e-5 --model_type advanced --vit_model vit_b_16 --pretrained 1