CDL-Net: A Lightweight Channel-Disentangled Network via Gradient Reversal for Facial Expression Recognition
This repository contains the official PyTorch implementation of the paper "CDL-Net: A Lightweight Channel-Disentangled Network via Gradient Reversal for Facial Expression Recognition".

Overview
CDL-Net is a lightweight facial expression recognition network designed for edge deployment. It explicitly splits the backbone features into an expression subspace and an identity subspace, utilizing a Gradient Reversal Layer (GRL) and an orthogonal loss to force the network to learn identity-invariant expression features.

Requirements
Python 3.8+

PyTorch >= 2.0.0

InsightFace (for offline pseudo-label extraction)

Install dependencies:
pip install -r requirements.txt

Data Preparation
Download the RAF-DB and ExpW datasets.

Align the face images to 224×224 and organize them into train and valid folders according to their class labels (0-6).

Extract identity pseudo-labels (if not already present in data/identity_vectors.pkl):
python utils/extract_identity.py

Training
To train CDL-Net on RAF-DB:
python train_cdlnet.py

Note: Please modify the DATA_ROOT and IDENTITY_FILE paths in train_cdlnet.py before running.

Note on Datasets and Weights
The RAF-DB and ExpW datasets are publicly available. Due to file size limits, the datasets and trained weight checkpoints (.pth) are not included in this repository.