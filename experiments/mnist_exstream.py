from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset

from experiments import Experiment, MNISTPretrained, MNISTPretrainedConfig
from models import ExStreamBuffer


@dataclass
class MNISTExStreamConfig(MNISTPretrainedConfig):
    pretrain_classes: list[int]
    stream_classes: list[int]
    stream_samples_per_class: int
    buffer_capacity: int
    stream_learning_rate: float
    eval_every: int
    use_replay: bool  # False gives the paper's "No Buffer" baseline


class MNISTExStream(MNISTPretrained):
    """Pretrains the CNN, freezes its convolutional layers, then keeps training the
    classifier from a stream of unseen samples, one sample at a time."""

    def __init__(self, config: MNISTExStreamConfig):
        """For type checking"""
        super().__init__(config)
        self.config = config
        self.stream_loader: DataLoader


    def init_dataloaders(self):
        full_train_dataset, full_test_dataset = MNISTPretrained._load_MNIST()
        pretrain_dataset = self._halve_MNIST(full_train_dataset, self.config.pretrain_classes)
        stream_dataset = self._select_stream(full_train_dataset, pretrain_dataset)

        # Only classes the model is expected to know by the end are tested
        known_classes = torch.tensor(self.config.pretrain_classes + self.config.stream_classes)
        test_indices = torch.where(torch.isin(full_test_dataset.targets, known_classes))[0]

        self.train_loader = DataLoader(
            pretrain_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        self.stream_loader = DataLoader(
            stream_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        self.test_loader = DataLoader(
            Subset(full_test_dataset, test_indices.tolist()),
            batch_size=self.config.batch_size,
            shuffle=False
        )


    def train_model(self):
        super().train_model()
        self._print_accuracy_per_class("Accuracy before streaming")

        for parameter in self.model.features.parameters():
            parameter.requires_grad_(False)

        buffer = None
        if self.config.use_replay:
            buffer = ExStreamBuffer(self.config.buffer_capacity)
            self._fill_buffer_from_pretraining_data(buffer)

        self._learn_from_stream(buffer)
        self._print_accuracy_per_class("Accuracy after streaming")


    def _select_stream(self, full_train_dataset, pretrain_dataset: Subset) -> Subset:
        """Picks samples for each stream class that pretraining never saw."""
        targets = full_train_dataset.targets
        unseen = torch.ones(len(targets), dtype=torch.bool)
        unseen[pretrain_dataset.indices] = False

        selected_indices = []
        for digit in self.config.stream_classes:
            digit_indices = torch.where((targets == digit) & unseen)[0]
            generator = torch.Generator().manual_seed(self.config.seed + digit)
            selected = digit_indices[
                torch.randperm(
                    len(digit_indices),
                    generator=generator
                )[:self.config.stream_samples_per_class]
            ]
            selected_indices.extend(selected.tolist())

        return Subset(full_train_dataset, selected_indices)


    def _fill_buffer_from_pretraining_data(self, buffer: ExStreamBuffer):
        """Old classes need prototypes too, or a new class would have nothing to replay against."""
        for images, labels in self.train_loader:
            for feature, label in zip(self._extract_features(images), labels.tolist()):
                buffer.add(feature, label)


    def _learn_from_stream(self, buffer: ExStreamBuffer | None):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.classifier.parameters(),
            lr=self.config.stream_learning_rate
        )

        seen = 0
        for images, labels in self.stream_loader:
            features = self._extract_features(images)
            labels = labels.to(Experiment.device)

            for feature, label in zip(features, labels):
                if buffer is None:
                    self._update_classifier(feature.unsqueeze(0), label.unsqueeze(0), criterion, optimizer)
                else:
                    buffer.add(feature, label.item())
                    self._rehearse(buffer, criterion, optimizer)

                seen += 1
                if seen % self.config.eval_every == 0:
                    print(f"    Stream samples seen: {seen} | Test Accuracy: {self.evaluate_model():.2f}%")


    def _rehearse(self, buffer: ExStreamBuffer, criterion: nn.modules.loss._Loss, optimizer: optim.Optimizer):
        """Replays every stored prototype once, in random order."""
        prototypes, labels = buffer.all_prototypes()
        order = torch.randperm(len(labels), device=labels.device)

        for batch in order.split(self.config.batch_size):
            self._update_classifier(prototypes[batch], labels[batch], criterion, optimizer)


    def _update_classifier(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.modules.loss._Loss,
        optimizer: optim.Optimizer
    ):
        optimizer.zero_grad()
        loss = criterion(self.model.classifier(features), labels)
        loss.backward()
        optimizer.step()


    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.features(images.to(Experiment.device)).flatten(1)


    def _print_accuracy_per_class(self, title: str):
        self.model.eval()

        correct = torch.zeros(10)
        total = torch.zeros(10)

        with torch.no_grad():
            for images, labels in self.test_loader:
                predictions = self.model(images.to(Experiment.device)).argmax(dim=1).cpu()

                for digit in labels.unique():
                    is_digit = labels == digit
                    correct[digit] += (predictions[is_digit] == digit).sum()
                    total[digit] += is_digit.sum()

        per_class = " | ".join(
            f"{digit}: {100 * correct[digit] / total[digit]:.2f}%"
            for digit in range(10) if total[digit] > 0
        )
        print(f"  {title}: {per_class}")
