from dataclasses import dataclass

import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from experiments import Experiment, Config, MnistPretrained


@dataclass
class MnistConfig(Config):
    batch_size: int
    learning_rate: float
    samples_per_class: int
    alpha: float
    pretrained: MnistPretrained
    all_classes: list[int] # all between 0 and 9
    streamed_classes: list[int] # all between 0 and 9, included in ``all_classes``


class MnistUsecase(Experiment):
    def __init__(self, config: MnistConfig):
        # For type checking
        super().__init__(config)
        self.config = config

        missing = set(config.streamed_classes) - set(config.all_classes)
        if missing:
            raise ValueError(
                "streamed_classes must be included in all_classes. "
                f"Missing: {sorted(missing)}"
            )


    def init_dataloaders(self):
        """Load MNIST and build the continual-learning training and test dataloaders."""
        full_train_dataset, full_test_dataset = self._load_mnist()
        train_dataset = self._training_superset(full_train_dataset)
        test_dataset = self._test_superset(full_test_dataset)

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


    def evaluate_model(self) -> float | dict[str, float]:
        self.model.eval()

        correct = {label: 0 for label in self.config.all_classes}
        total = {label: 0 for label in self.config.all_classes}

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(Experiment.device)
                labels = labels.to(Experiment.device)

                outputs = self.model(images)
                predictions = outputs.argmax(dim=1)

                for index, label in enumerate(self.config.all_classes):
                    mask = labels == index
                    correct[label] += (predictions[mask] == labels[mask]).sum().item()
                    total[label] += mask.sum().item()

        accuracy = {
            str(label): 100 * correct[label] / total[label]
            for label in self.config.all_classes
            if total[label] > 0
        }

        nb_correct = sum(correct.values())
        nb_total = sum(total.values())
        accuracy["global"] = 100 * nb_correct / nb_total

        return accuracy


    def _load_mnist(self) -> tuple[datasets.MNIST, datasets.MNIST]:
        """Load the MNIST datasets with digit-to-model-index remapping."""
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
        """Map the model's classes to their contiguous output indices."""
        return {
            label: index
            for index, label in enumerate(self.config.all_classes)
        }


    def _training_superset(self, full_train_dataset: datasets.MNIST) -> Subset:
        """Select training samples for the configured streamed classes."""
        # IMPORTANT: use the part of MNIST that hasn't been used to train the pretrained model
        targets = full_train_dataset.targets
        pretrained_classes = self.config.pretrained.config.classes_kept

        # Indices already consumed by the pretrained experiment are excluded
        # for the old classes. New classes were never seen before, so every
        # sample is available for them
        unused_indices = self._unused_pretrained_indices(targets)

        selected_indices = []
        for digit in self.config.streamed_classes:
            if digit in pretrained_classes:
                candidates = unused_indices[targets[unused_indices] == digit]
            else:
                candidates = torch.where(targets == digit)[0]

            count = min(self.config.samples_per_class, len(candidates))
            generator = torch.Generator().manual_seed(self.config.seed + digit)
            permutation = torch.randperm(len(candidates), generator=generator)
            selected_indices.extend(
                candidates[permutation[:count]].tolist()
            )

        return Subset(full_train_dataset, selected_indices)


    def _unused_pretrained_indices(self, targets: torch.Tensor) -> torch.Tensor:
        """Recompute the pretrained samples deterministically.

        The pretrained experiment splits each kept digit with a generator
        seeded by ``seed + digit``; the unused part is everything after the
        first ``samples_per_class`` samples.
        """
        config = self.config.pretrained.config
        unused_indices = []

        for digit in config.classes_kept:
            digit_indices = torch.where(targets == digit)[0]
            generator = torch.Generator().manual_seed(config.seed + digit)
            permutation = torch.randperm(len(digit_indices), generator=generator)
            unused_indices.extend(
                digit_indices[permutation[config.samples_per_class:]].tolist()
            )

        return torch.tensor(unused_indices, dtype=torch.long)


    def _test_superset(self, full_test_dataset: datasets.MNIST) -> Subset:
        """Select test samples for the evaluated ``all_classes`` only."""
        test_indices = [
            index
            for index, label in enumerate(full_test_dataset.targets.tolist())
            if label in self.config.all_classes
        ]

        return Subset(full_test_dataset, test_indices)
