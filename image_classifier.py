"""Image classification with transfer learning and a custom PyTorch training loop.

Uses a pretrained ResNet18 backbone (transfer learning), a custom training loop
covering forward pass, loss, backpropagation, validation, early stopping, and
model checkpointing.

Run:  python image_classifier.py
"""
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from config import DEVICE, IMG_MODEL_PATH, IMG_BATCH_SIZE, IMG_EPOCHS, IMG_LR


def build_dataloaders(batch_size: int = IMG_BATCH_SIZE):
    """Load CIFAR-10 with standard augmentation for training."""
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def build_model(num_classes: int = 10) -> nn.Module:
    """ResNet18 with a fresh classification head (transfer learning)."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze the backbone; train only the new head.
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over the data. If optimizer is None, runs in eval mode."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)              # forward pass
            loss = criterion(outputs, labels)    # loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()                  # backpropagation
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train(epochs: int = IMG_EPOCHS, patience: int = 2):
    """Custom training loop with validation, early stopping, and checkpointing."""
    train_loader, val_loader = build_dataloaders()
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=IMG_LR)

    best_acc, best_weights, epochs_no_improve = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch}/{epochs} | "
              f"train_loss={train_loss:.3f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.3f} acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc, best_weights = val_acc, copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_weights:
        model.load_state_dict(best_weights)
        torch.save(model.state_dict(), IMG_MODEL_PATH)
        print(f"Best val accuracy: {best_acc:.3f}. Saved to {IMG_MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()
