import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os

from models.disentangle_net import DisentangleNet
from utils.dataset import RAFDBDisentangleDataset

# ================= Config =================
BATCH_SIZE = 32
EPOCHS = 30
WARMUP_EPOCHS = 15
LR = 1e-4
ALPHA, BETA, GAMMA = 1.0, 0.5, 0.1
EXPR_CHANNELS = 288
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 请在这里填入你的本地数据集路径
DATA_ROOT = "./datasets/RAF-DB" 
IDENTITY_FILE = "./data/identity_vectors.pkl"
# ==========================================

def cosine_similarity_loss(a, b):
    if a.size(1) != b.size(1):
        target_dim = min(a.size(1), b.size(1))
        a = torch.nn.functional.adaptive_avg_pool1d(a.unsqueeze(1), target_dim).squeeze(1)
        b = torch.nn.functional.adaptive_avg_pool1d(b.unsqueeze(1), target_dim).squeeze(1)
    a_norm = nn.functional.normalize(a, dim=1)
    b_norm = nn.functional.normalize(b, dim=1)
    return (a_norm * b_norm).sum(dim=1).abs().mean()

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def train():
    train_dataset = RAFDBDisentangleDataset(split='train', data_root=DATA_ROOT, identity_file=IDENTITY_FILE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = DisentangleNet(num_classes=7, use_grl=True, expr_channels=EXPR_CHANNELS).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    class_counts = [1290, 281, 717, 4772, 1982, 705, 2524]
    class_weights = 1.0 / torch.sqrt(torch.tensor(class_counts, dtype=torch.float32))
    class_weights = class_weights / class_weights.sum() * 7
    criterion_weighted = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.1)
    criterion_plain = nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion_ident = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        # 两阶段切换逻辑
        criterion_expr = criterion_weighted if epoch < WARMUP_EPOCHS else criterion_plain
        
        for step, (images, labels, identity_vecs) in enumerate(train_loader):
            images, labels, identity_vecs = images.to(DEVICE), labels.to(DEVICE), identity_vecs.to(DEVICE)
            
            # Mixup 数据增强
            use_mix = np.random.rand() < 0.5
            if use_mix:
                images, labels_a, labels_b, lam = mixup_data(images, labels, 0.2)

            optimizer.zero_grad()
            with autocast():
                expr_out, ident_out, expr_feat, ident_feat = model(images)
                if use_mix:
                    loss_expr = lam * criterion_expr(expr_out, labels_a) + (1 - lam) * criterion_expr(expr_out, labels_b)
                else:
                    loss_expr = criterion_expr(expr_out, labels)
                
                loss_ident = criterion_ident(ident_out, identity_vecs)
                loss_ortho = cosine_similarity_loss(expr_feat, ident_feat)
                loss = ALPHA * loss_expr + BETA * loss_ident + GAMMA * loss_ortho

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        print(f"Epoch {epoch+1} done.")

if __name__ == '__main__':
    train()