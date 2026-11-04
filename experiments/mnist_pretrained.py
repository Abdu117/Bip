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
    learning_rate: float
    classes_kept: list[int] # all between 0 and 9


class MNISTPretrained(Experiment):
    def __init__(self, config: MNISTPretrainedConfig):
        # For type checking
        super().__init__(config)
        self.config = config

        self.unused_dataset: Subset


    def init_dataloaders(self):
        full_train_dataset, full_test_dataset = self._load_MNIST()
        train_dataset = self._training_subset(full_train_dataset)
        test_dataset = self._test_subset(full_test_dataset)

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


    def _load_MNIST(self) -> tuple[datasets.MNIST, datasets.MNIST]:
        transform = transforms.ToTensor()
        label_to_index = self._label_to_index()
        target_transform = lambda label: label_to_index[label]

        full_train_dataset = datasets.MNIST(
            root="./.data",
            train=True,
            download=True,
            transform=transform,
            target_transform=target_transform
        )

        full_test_dataset = datasets.MNIST(
            root="./.data",
            train=False,
            download=True,
            transform=transform,
            target_transform=target_transform
        )

        return full_train_dataset, full_test_dataset


    def _label_to_index(self) -> dict[int, int]:
        """Map kept MNIST digits to contiguous class indices (0..n-1)}. Used for subset setting
        Example: {0, 3, 5} become {0: 0, 1: 3, 2: 5}"""
        return {
            label: index
            for index, label in enumerate(self.config.classes_kept)
        }


    def _training_subset(self, full_train_dataset: datasets.MNIST) -> Subset:
        selected_indices = []
        unused_indices = []

        targets = full_train_dataset.targets

        for digit in self.config.classes_kept:
            digit_indices = torch.where(targets == digit)[0]

            # Randomly select images from this digit
            generator = torch.Generator().manual_seed(self.config.seed + digit)
            permutation = torch.randperm(
                len(digit_indices),
                generator=generator
            )

            selected = digit_indices[
                permutation[:self.config.samples_per_class]
            ]
            selected_indices.extend(selected.tolist())

            # Store unused indices for experiments that extend the pretrained
            unused = digit_indices[
                permutation[self.config.samples_per_class:]
            ]
            unused_indices.extend(unused.tolist())

        train_dataset = Subset(
            full_train_dataset,
            selected_indices
        )

        self.unused_dataset = Subset(
            full_train_dataset,
            unused_indices
        )

        return train_dataset


    def _test_subset(self, full_test_dataset: datasets.MNIST) -> Subset:
        label_to_index = self._label_to_index()
        test_indices = [
            index
            for index, label in enumerate(full_test_dataset.targets.tolist())
            if label in label_to_index
        ]

        test_subset = Subset(full_test_dataset, test_indices)
        return test_subset


    def _init_training_objects(self) -> tuple[nn.Module, nn.modules.loss._Loss, optim.Optimizer]:
        model = CNN(len(self.config.classes_kept)).to(Experiment.device)

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
