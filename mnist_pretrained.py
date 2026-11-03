import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import numpy as np
import random

# ==========================================
# 1. Configuration
# ==========================================

SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
SAMPLES_PER_CLASS = 3000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================
# 2. Load MNIST
# ==========================================

transform = transforms.ToTensor()

full_train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# ==========================================
# 3. Select 50% of training data
# ==========================================

selected_indices = []

targets = full_train_dataset.targets

for digit in range(10):

    digit_indices = torch.where(targets == digit)[0]

    # Randomly select 3000 images from this digit
    generator = torch.Generator().manual_seed(SEED + digit)

    selected = digit_indices[
        torch.randperm(
            len(digit_indices),
            generator=generator
        )[:SAMPLES_PER_CLASS]
    ]

    selected_indices.extend(selected.tolist())

train_dataset = Subset(
    full_train_dataset,
    selected_indices
)

print("Training samples:", len(train_dataset))
print("Test samples:", len(test_dataset))

# 30,000 training images = 50% of MNIST training set


# ==========================================
# 4. Create DataLoaders
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 5. Define CNN
# ==========================================

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


model = CNN().to(DEVICE)


# ==========================================
# 6. Loss and Optimizer
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# 7. Training Function
# ==========================================

def train_one_epoch(model, loader):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader)

    return avg_loss, accuracy


# ==========================================
# 8. Evaluation Function
# ==========================================

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    return accuracy


# ==========================================
# 9. Train for 10 Epochs
# ==========================================

for epoch in range(EPOCHS):

    loss, train_accuracy = train_one_epoch(
        model,
        train_loader
    )

    test_accuracy = evaluate(
        model,
        test_loader
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )


# ==========================================
# 10. Save Model
# ==========================================

torch.save(
    model.state_dict(),
    "mnist_pretrained.pth"
)

print("\nPretrained model saved successfully!")

print("Final test accuracy:", evaluate(model, test_loader))