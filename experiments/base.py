"""
experiments/base.py
===================
The generic skeleton that every experiment in this project is built on.

Design idea (the "template method" pattern):
    `Experiment.conduct_full()` defines the ORDER of the steps that every experiment goes
    through (seed -> load data -> train -> save -> evaluate). The steps that differ between
    experiments (`init_dataloaders`, `train_model`, `evaluate_model`) are left empty here
    and are implemented by subclasses, e.g. MNISTPretrained. To add a new experiment
    (for example continual learning with EWC) you write a new subclass and only override
    those steps; the surrounding pipeline is reused unchanged.
"""
from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader


@dataclass
class Config:
    """Settings that every experiment needs. Subclasses add their own fields.

    @dataclass automatically generates __init__, __repr__ etc. from the annotated fields,
    so `Config(name=..., seed=..., model_weights_filepath=...)` just works.
    """
    name: str                       # label shown in the log output
    seed: int                       # seed for all random number generators
    model_weights_filepath: Path    # where save_model() writes the trained weights


class Experiment:
    """Base class of all experiments. Not meant to be run directly."""

    # Class-level (shared by all experiments in the running program):
    # remembers whether the device has already been chosen, so it happens only once.
    torch_initialized: bool = False
    # The device ("cuda" = GPU or "cpu") that tensors and the model are moved to.
    # Only declared here; it gets its value in init_torch(). Accessed everywhere as
    # `Experiment.device`.
    device: torch.device

    def __init__(self, config: Config):
        # The lines below only DECLARE which attributes will exist (type annotations
        # without a value). They are assigned later by the subclass:
        self.train_loader: DataLoader           # feeds training batches (set in init_dataloaders)
        self.test_loader: DataLoader            # feeds test batches     (set in init_dataloaders)
        self.val_loader: DataLoader | None = None   # optional validation data; None = not used
        self.model: torch.nn.Module             # the network (set in train_model)

        self.config = config

    # ------------------------------------------------------------------
    # Steps that subclasses MUST implement.
    # Note: Experiment does not inherit from abc.ABC, so Python does not stop you from
    # instantiating it; the NotImplementedError is the safeguard if a step is forgotten.
    # ------------------------------------------------------------------
    @abstractmethod
    def init_dataloaders(self) -> None:
        """Create self.train_loader and self.test_loader (and optionally self.val_loader)."""
        raise NotImplementedError()

    @abstractmethod
    def train_model(self) -> None:
        """Create/load self.model and train it."""
        raise NotImplementedError()

    @abstractmethod
    def evaluate_model(self) -> float:
        """Return the accuracy of self.model on the test data, in percent."""
        raise NotImplementedError()


    @staticmethod
    def init_torch():
        """Choose the compute device once: the GPU if CUDA is available, otherwise the CPU."""
        # Stored on the CLASS (Experiment.device), so all experiments share it.
        Experiment.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Using device:", Experiment.device)


    def conduct_full(self):
        """Run the complete experiment. This is the method you call from main.py."""
        # Choose the device only for the first experiment of the program run.
        if not Experiment.torch_initialized:
            Experiment.init_torch()
            Experiment.torch_initialized = True

        # Seed first, so that everything that follows (data selection, weight
        # initialisation, shuffling) is reproducible.
        self.seed_environment()
        print(f"Starting experiment '{self.config.name}'. Environment seed: {self.config.seed}.")

        print("  Loading datasets...")
        self.init_dataloaders()
        # len(loader.dataset) = number of images (len(loader) would be the number of batches).
        print("  Training samples:", len(self.train_loader.dataset))
        print("  Test samples:", len(self.test_loader.dataset))
        if self.val_loader is not None:
            print("  Validation samples:", len(self.val_loader.dataset))

        print("  Training model...")
        self.train_model()

        print("  Saving model...")
        self.save_model()
        print("  Model saved successfully!")
        # One more evaluation with the final weights -> the "Final test accuracy" line.
        print(f"  Final test accuracy: {self.evaluate_model()}%")


    def seed_environment(self):
        """Reproducibility: seed every random number generator that is used.

        With the same seed, repeated runs select the same data, start from the same
        initial weights and shuffle in the same order (on the same hardware/software;
        GPU runs can still differ slightly in the last decimals).
        """
        seed = self.config.seed
        torch.manual_seed(seed)     # PyTorch (weight init, DataLoader shuffling, ...)
        np.random.seed(seed)        # NumPy
        random.seed(seed)           # Python's built-in random module

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)    # PyTorch on all GPUs


    def save_model(self):
        """Save the learned weights (a "state_dict") to config.model_weights_filepath."""
        path = self.config.model_weights_filepath
        # Create the target folder (e.g. .out) if it does not exist yet. Without
        # parents=True this only works when the folder's own parent already exists.
        path.parent.mkdir(exist_ok=True)
        # Only the weights are stored, NOT the architecture. To load them later you must
        # first create a CNN() and then call model.load_state_dict(torch.load(path)).
        torch.save(
            self.model.state_dict(),
            path
        )