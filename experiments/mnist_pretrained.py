from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from experiments import Experiment, Config
from models import CNN


@dataclass
class MNISTPretrainedConfig(Config):
    batch_size: int
    epochs: int
    samples_per_class: int
    learning_rate: int


class MNISTPretrained(Experiment):
    def __init__(self, config: MNISTPretrainedConfig):
        """For type checking"""
        super().__init__(config)
        self.config = config


    def init_dataloaders(self):
        full_train_dataset, test_dataset = MNISTPretrained._load_MNIST()
        train_dataset = self._halve_MNIST(full_train_dataset)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )


    def train_model(self):
        self.model, criterion, optimizer = self._init_training_objects()

        for epoch in range(self.config.epochs):
            loss, train_accuracy = self._train_one_epoch(criterion, optimizer)

            test_accuracy = self.evaluate_model()
            print(
                f"    Epoch [{epoch + 1}/{self.config.epochs}] "
                f"Loss: {loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.2f}% | "
                f"Test Accuracy: {test_accuracy:.2f}%"
            )


    def evaluate_model(self) -> float:
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(Experiment.device)
                labels = labels.to(Experiment.device)

                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = 100 * correct / total
        return accuracy


    @staticmethod
    def _load_MNIST() -> tuple[datasets.MNIST, datasets.MNIST]:
        transform = transforms.ToTensor()

        full_train_dataset = datasets.MNIST(
            root="./.data",
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.MNIST(
            root="./.data",
            train=False,
            download=True,
            transform=transform
        )

        return full_train_dataset, test_dataset


    def _halve_MNIST(self, full_train_dataset: datasets.MNIST, classes: Iterable[int] = range(10)) -> Subset:
        selected_indices = []

        targets = full_train_dataset.targets

        for digit in classes:
            digit_indices = torch.where(targets == digit)[0]

            # Randomly select 3000 images from this digit
            generator = torch.Generator().manual_seed(self.config.seed + digit)

            selected = digit_indices[
                torch.randperm(
                    len(digit_indices),
                    generator=generator
                )[:self.config.samples_per_class]
            ]

            selected_indices.extend(selected.tolist())

        train_dataset = Subset(
            full_train_dataset,
            selected_indices
        )

        return train_dataset


    def _init_training_objects(self) -> tuple[nn.Module, nn.modules.loss._Loss, optim.Optimizer]:
        model = CNN().to(Experiment.device)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate
        )

        return model, criterion, optimizer


    def _train_one_epoch(self, criterion: nn.modules.loss._Loss, optimizer: optim.Optimizer) -> tuple[float, float]:
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(Experiment.device)
            labels = labels.to(Experiment.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(self.train_loader)

        return avg_loss, accuracy
