from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from experiments import Experiment, Config, MnistPretrained
from models import Cnn, Der


@dataclass
class MnistDerConfig(Config):
    batch_size: int
    evals: int
    learning_rate: float
    samples_per_class: int
    alpha: float
    pretrained: MnistPretrained
    classes_added: list[int] # all between 0 and 9
    buffer_size: int = 500


class MnistDer(Experiment):
    def __init__(self, config: MnistDerConfig):
        # For type checking
        super().__init__(config)
        self.config = config
        self.model: Der

        # Each eval consumes a different, non-overlapping chunk of the
        # training set, to simulate a stream of captured data
        self.eval_train_indices: list[list[int]]


    def init_dataloaders(self):
        full_train_dataset, full_test_dataset = self._load_mnist()
        train_dataset = self._training_superset(full_train_dataset)
        test_dataset = self._test_superset(full_test_dataset)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        self.eval_train_indices = self._split_training_indices(
            len(train_dataset)
        )


    def train_model(self):
        self.model, criterion, optimizer = self._init_training_objects()
        print(f"  Initial test accuracy: {self.evaluate_model()}%")

        for eval in range(self.config.evals):
            loss, train_accuracy = self._train_one_eval(
                criterion, optimizer, eval
            )

            test_accuracy = self.evaluate_model()
            print(
                f"    Eval [{eval + 1}/{self.config.evals}] "
                f"Loss: {loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.2f}% | "
                f"Test Accuracy: {test_accuracy:.2f}% | "
                f"Buffer: {self.model.buffer_length}/{self.model.buffer_size}"
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


    def _split_training_indices(self, nb_samples: int) -> list[list[int]]:
        """Split the training set into one disjoint random chunk per eval."""
        generator = torch.Generator().manual_seed(self.config.seed)
        permutation = torch.randperm(nb_samples, generator=generator).tolist()

        return [
            permutation[
                nb_samples * eval // self.config.evals:
                nb_samples * (eval + 1) // self.config.evals
            ]
            for eval in range(self.config.evals)
        ]


    def _all_classes(self) -> list[int]:
        """Classes of the DER model, in output-index order.

        The pretrained classes come first so their output indices are
        preserved when the classifier head is expanded.
        """
        classes = self.config.pretrained.config.classes_kept + self.config.classes_added
        return list(dict.fromkeys(classes))


    def _load_mnist(self) -> tuple[datasets.MNIST, datasets.MNIST]:
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
        """Map MNIST digits to the model's contiguous class indices."""
        return {
            label: index
            for index, label in enumerate(self._all_classes())
        }


    def _training_superset(self, full_train_dataset: datasets.MNIST) -> Subset:
        # IMPORTANT: use the part of MNIST that hasn't been used to train the pretrained model
        targets = full_train_dataset.targets
        pretrained_classes = self.config.pretrained.config.classes_kept

        # Indices already consumed by the pretrained experiment are excluded
        # for the old classes. New classes were never seen before, so every
        # sample is available for them
        unused_indices = torch.tensor(
            self.config.pretrained.unused_dataset.indices,
            dtype=torch.long
        )

        selected_indices = []
        for digit in self._all_classes():
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


    def _test_superset(self, full_test_dataset: datasets.MNIST) -> Subset:
        label_to_index = self._label_to_index()
        test_indices = [
            index
            for index, label in enumerate(full_test_dataset.targets.tolist())
            if label in label_to_index
        ]

        return Subset(full_test_dataset, test_indices)


    def _init_training_objects(self) -> tuple[nn.Module, nn.modules.loss._Loss, optim.Optimizer]:
        backbone = Cnn.from_cnn(
            self.config.pretrained.model,
            len(self._all_classes())
        ).to(Experiment.device)

        model = Der(backbone, self.config.buffer_size).to(Experiment.device)

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate
        )

        return model, criterion, optimizer


    def _train_one_eval(self, criterion: nn.modules.loss._Loss, optimizer: optim.Optimizer, eval: int) -> tuple[float, float]:
        self.model.train()

        eval_dataset = Subset(
            self.train_loader.dataset,
            self.eval_train_indices[eval]
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        if len(eval_loader) == 0:
            return 0.0, 0.0

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in eval_loader:
            images = images.to(Experiment.device)
            labels = labels.to(Experiment.device)

            optimizer.zero_grad()

            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # Match the logits the model produced on
            # buffered samples when they were first seen
            replay = self.model.sample_buffer(images.size(0))
            if replay is not None:
                replay_images, _, replay_logits = replay
                current_replay_logits = self.model(replay_images)
                replay_loss = F.mse_loss(current_replay_logits, replay_logits)
                loss = loss + self.config.alpha * replay_loss

            loss.backward()
            optimizer.step()

            # Store the logits computed before the update
            self.model.add_to_buffer(images, labels, outputs.detach())

            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(eval_loader)

        return avg_loss, accuracy
