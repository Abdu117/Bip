"""
experiments/mnist_pretrained.py
===============================
Experiment: pretrain the CNN on HALF of the MNIST training set.

This is step 1 of the continual-learning use case. The model learns all 10 digits from
30,000 images (3,000 per class) and is then evaluated on the 10,000 official test images.
The saved weights are the starting point for the continual-learning step.
"""
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
    """Config = the base fields (name, seed, model_weights_filepath) plus these:"""
    batch_size: int         # images per optimisation step
    epochs: int             # number of full passes over the training data
    samples_per_class: int  # images per digit that go into the training half
    learning_rate: int      # step size of Adam (type hint says int, but a float like
                            # 0.001 is what is actually passed; Python does not enforce
                            # type hints, so it works, it should just read `float`)


class MNISTPretrained(Experiment):
    def __init__(self, config: MNISTPretrainedConfig):
        """For type checking"""
        # The parent __init__ already stores the config, but it is typed as the generic
        # `Config`. Assigning it again here lets editors/type checkers know it is the more
        # specific MNISTPretrainedConfig (with batch_size, epochs, ...).
        super().__init__(config)
        self.config = config


    # ==================================================================
    # Step 1: data
    # ==================================================================
    def init_dataloaders(self):
        # Load the full MNIST: 60,000 training images and 10,000 test images.
        full_train_dataset, test_dataset = MNISTPretrained._load_MNIST()
        # Reduce the training images to the 30,000-image half (3,000 per digit).
        train_dataset = self._halve_MNIST(full_train_dataset)

        # A DataLoader cuts a dataset into batches and delivers them one by one.
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True        # new random order every epoch -> better training
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False       # order is irrelevant for evaluation
        )


    # ==================================================================
    # Step 2: training
    # ==================================================================
    def train_model(self):
        # Creates a NEW, randomly initialised CNN (this is where self.model is set).
        self.model, criterion, optimizer = self._init_training_objects()

        for epoch in range(self.config.epochs):
            # One full pass over the training data. The weights are updated in here.
            loss, train_accuracy = self._train_one_epoch(criterion, optimizer)

            # Measure the model on the unseen test images after every epoch, to watch
            # how well it generalises. This does NOT influence the training.
            test_accuracy = self.evaluate_model()
            print(
                f"    Epoch [{epoch + 1}/{self.config.epochs}] "
                f"Loss: {loss:.4f} | "
                f"Train Accuracy: {train_accuracy:.2f}% | "
                f"Test Accuracy: {test_accuracy:.2f}%"
            )


    # ==================================================================
    # Step 3: evaluation (also called once more at the end by conduct_full)
    # ==================================================================
    def evaluate_model(self) -> float:
        # eval() switches layers like Dropout/BatchNorm to inference behaviour. This CNN
        # has neither, so it changes nothing here, but it is good practice.
        self.model.eval()

        correct = 0
        total = 0

        # no_grad(): don't build the computation graph / gradients. Saves memory and time,
        # and makes sure nothing is learned during evaluation.
        with torch.no_grad():
            for images, labels in self.test_loader:
                # Move the batch to the same device (CPU/GPU) as the model.
                images = images.to(Experiment.device)
                labels = labels.to(Experiment.device)

                outputs = self.model(images)            # (batch, 10) scores
                predictions = outputs.argmax(dim=1)     # index of the highest score = predicted digit

                # (predictions == labels) is a True/False tensor; sum() counts the Trues;
                # .item() converts the one-element tensor into a normal Python number.
                correct += (predictions == labels).sum().item()
                total += labels.size(0)                 # number of images in this batch

        accuracy = 100 * correct / total    # percentage of correctly classified test images
        return accuracy


    # ==================================================================
    # Helper methods (the leading underscore means "internal, not for outside use")
    # ==================================================================
    @staticmethod
    def _load_MNIST() -> tuple[datasets.MNIST, datasets.MNIST]:
        # Converts each image (28x28 grey values 0-255) into a float tensor of shape
        # (1, 28, 28) with values scaled to 0.0-1.0. No further normalisation is applied.
        transform = transforms.ToTensor()

        full_train_dataset = datasets.MNIST(
            root="./.data",     # download folder (relative to where you start the script)
            train=True,         # the 60,000 training images
            download=True,      # download only if not already present
            transform=transform
        )

        test_dataset = datasets.MNIST(
            root="./.data",
            train=False,        # the 10,000 official test images
            download=True,
            transform=transform
        )

        return full_train_dataset, test_dataset


    def _halve_MNIST(self, full_train_dataset: datasets.MNIST) -> Subset:
        """Pick `samples_per_class` random images of every digit (a balanced half).

        The choice is deterministic: the same seed and samples_per_class always give the
        same images. That is what makes it possible to later reconstruct "the other half".
        """
        selected_indices = []

        # All 60,000 labels as one tensor, e.g. tensor([5, 0, 4, 1, 9, ...]).
        targets = full_train_dataset.targets

        for digit in range(10):
            # Positions (indices) of all training images that show this digit.
            digit_indices = torch.where(targets == digit)[0]

            # A separate random generator per digit, seeded with seed + digit, so the
            # selection does not depend on any other random activity in the program.
            # Randomly select 3000 images from this digit
            generator = torch.Generator().manual_seed(self.config.seed + digit)

            # randperm = random permutation of 0..n-1; take the first samples_per_class
            # of them and use them to pick entries out of digit_indices.
            selected = digit_indices[
                torch.randperm(
                    len(digit_indices),
                    generator=generator
                )[:self.config.samples_per_class]
            ]

            selected_indices.extend(selected.tolist())

        # A Subset is a "view" onto the full dataset that exposes only the chosen indices.
        # No images are copied. The chosen indices stay available as `train_dataset.indices`.
        train_dataset = Subset(
            full_train_dataset,
            selected_indices
        )

        return train_dataset


    def _init_training_objects(self) -> tuple[nn.Module, nn.modules.loss._Loss, optim.Optimizer]:
        # A new CNN with random initial weights (reproducible thanks to the seed), moved
        # to the GPU if one is available.
        model = CNN().to(Experiment.device)

        # Measures how wrong the predictions are. For classification with raw scores this
        # is: softmax -> probability of the correct digit -> negative log of it.
        criterion = nn.CrossEntropyLoss()

        # Decides HOW the weights are changed using the gradients. Adam adapts the step
        # size per weight, which makes it work well without much tuning.
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate
        )

        return model, criterion, optimizer


    def _train_one_epoch(self, criterion: nn.modules.loss._Loss, optimizer: optim.Optimizer) -> tuple[float, float]:
        """One pass over all training batches. Returns (average loss, train accuracy in %)."""
        # train() = training mode (the counterpart of eval()).
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(Experiment.device)
            labels = labels.to(Experiment.device)

            # PyTorch ADDS new gradients to old ones, so they must be reset for every batch.
            optimizer.zero_grad()
            # Forward pass: the model's scores for this batch, shape (batch, 10).
            outputs = self.model(images)
            # How wrong are those scores compared with the true labels?
            loss = criterion(outputs, labels)
            # Backward pass: compute d(loss)/d(weight) for every weight.
            loss.backward()
            # Update step: change every weight a little in the direction that lowers the loss.
            optimizer.step()

            # ---- Statistics only (they do not influence the learning) ----
            # .item() turns the loss tensor into a plain float.
            total_loss += loss.item()
            # The digit the model would currently choose for each image.
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        # Because the weights change after every batch, this accuracy is an average over a
        # model that is still improving during the epoch (so it is usually lower than the
        # test accuracy measured at the end of the epoch, especially in epoch 1).
        accuracy = 100 * correct / total
        # Mean loss per batch over the epoch.
        avg_loss = total_loss / len(self.train_loader)

        return avg_loss, accuracy