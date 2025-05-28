import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from math import log10
from model import EnhancedUNet
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean

class DegradationDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir, id_list, transform=None):
        self.degraded_dir = degraded_dir
        self.clean_dir = clean_dir
        self.id_list = id_list
        self.transform = transform

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        img_id = self.id_list[idx]
        degraded_path = os.path.join(self.degraded_dir, f"{img_id}.png")

        if img_id.startswith("rain"):
            clean_id = img_id.replace("rain", "rain_clean")
        else:
            clean_id = img_id.replace("snow", "snow_clean")

        clean_path = os.path.join(self.clean_dir, f"{clean_id}.png")

        degraded = Image.open(degraded_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')

        if self.transform:
            degraded = self.transform(degraded)
            clean = self.transform(clean)

        return degraded, clean

def compute_psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction='none')
    mse = mse.view(mse.shape[0], -1).mean(dim=1)
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.mean().item()

def train(model, dataloader, optimizer, device, criterion, scaler, accumulation_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for i, (degraded, clean) in enumerate(tqdm(dataloader)):
        degraded, clean = degraded.to(device), clean.to(device)
        with torch.cuda.amp.autocast():
            output = model(degraded)
            loss = criterion(output, clean) / accumulation_steps
        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        running_loss += loss.item() * degraded.size(0) * accumulation_steps
    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, device):
    model.eval()
    psnr_total = 0.0
    with torch.no_grad():
        for degraded, clean in dataloader:
            degraded, clean = degraded.to(device), clean.to(device)
            output = model(degraded)
            psnr_total += compute_psnr(output, clean) * degraded.size(0)
    return psnr_total / len(dataloader.dataset)

if __name__ == '__main__':
    num_epochs = 50
    batch_size = 2
    learning_rate = 2e-4
    accumulation_steps = 8

    transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(15),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
        T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.8),
        T.RandomApply([T.RandomAffine(0, translate=(0.1, 0.1))], p=0.3),
        T.ToTensor(),
        T.RandomApply([AddGaussianNoise(0., 0.02)], p=0.3),
        T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])

    degraded_dir = 'data/train/degraded'
    clean_dir = 'data/train/clean'

    rain_ids = [f"rain-{i}" for i in range(1, 1601)]
    snow_ids = [f"snow-{i}" for i in range(1, 1601)]
    all_ids = rain_ids + snow_ids
    train_ids, val_ids = train_test_split(all_ids, test_size=200, random_state=42)

    train_dataset = DegradationDataset(degraded_dir, clean_dir, train_ids, transform)
    val_dataset = DegradationDataset(degraded_dir, clean_dir, val_ids, transform=T.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=5e-6
)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    val_psnrs = []

    best_psnr = 0.0
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device, criterion, scaler, accumulation_steps)
        val_psnr = validate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val PSNR: {val_psnr:.2f} dB")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved at epoch {epoch+1} with PSNR: {val_psnr:.2f} dB")

    # Plot after training
    epochs = range(1, num_epochs + 1)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss', color=color)
    ax1.plot(epochs, train_losses, marker='o', color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Val PSNR (dB)', color=color)
    ax2.plot(epochs, val_psnrs, marker='x', color=color, label='Val PSNR')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Training Loss and Validation PSNR')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig('train_val_curve.png')
    plt.show()
