from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from experiments import Experiment, Config
from models import MnistCnn


@dataclass
class MnistPretrainedConfig(Config):
    batch_size: int
    epochs: int
    samples_per_class: int
    learning_rate: float
    classes_kept: list[int] # all between 0 and 9


class MnistPretrained(Experiment):
    def __init__(self, config: MnistPretrainedConfig):
        # For type checking and Experiment.from_kwargs
        super().__init__(config)
        self.config = config


    def init_dataloaders(self):
        """Load MNIST and build the pretrained training and test dataloaders."""
        full_train_dataset, full_test_dataset = self._load_mnist()
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


    def init_model(self) -> nn.Module:
        """Create a CNN with one output per kept class."""
        return MnistCnn(self.config.classes_kept).to(Experiment.device)


    def train_model(self):
        """Train the CNN for the configured epochs and log each epoch's measures."""
        self.model, criterion, optimizer = self._init_training_objects()

        for epoch in range(self.config.epochs):
            loss, train_accuracy = self._train_one_epoch(criterion, optimizer)

            test_accuracy = self.evaluate_model()

            self.log_measure(
                epoch=epoch + 1,
                loss=loss,
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy
            )


    def evaluate_model(self) -> float | dict[str, float]:
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


    def _load_mnist(self) -> tuple[datasets.MNIST, datasets.MNIST]:
        """Load the MNIST train and test datasets with digit-to-index remapping."""
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
        """Map kept MNIST digits to contiguous model class indices.

        For example, {0, 3, 5} become {0: 0, 3: 1, 5: 2}.
        """
        return {
            label: index
            for index, label in enumerate(self.config.classes_kept)
        }


    def _training_subset(self, full_train_dataset: datasets.MNIST) -> Subset:
        """Select a deterministic sample subset for each kept class."""
        selected_indices = []

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

        train_dataset = Subset(
            full_train_dataset,
            selected_indices
        )

        return train_dataset


    def _test_subset(self, full_test_dataset: datasets.MNIST) -> Subset:
        """Select the test samples belonging to the kept classes."""
        label_to_index = self._label_to_index()
        test_indices = [
            index
            for index, label in enumerate(full_test_dataset.targets.tolist())
            if label in label_to_index
        ]

        test_subset = Subset(full_test_dataset, test_indices)
        return test_subset


    def _init_training_objects(self) -> tuple[nn.Module, nn.modules.loss._Loss, optim.Optimizer]:
        """Create the model, loss function and optimizer for training."""
        model = self.init_model()

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate
        )

        return model, criterion, optimizer


    def _train_one_epoch(self, criterion: nn.modules.loss._Loss, optimizer: optim.Optimizer) -> tuple[float, float]:
        """Train for one epoch and return the average loss and training accuracy."""
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
