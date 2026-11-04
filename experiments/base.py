from abc import abstractmethod
from dataclasses import dataclass
import numpy as np
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader


@dataclass
class Config:
    name: str
    seed: int
    model_weights_filepath: Path


class Experiment:
    torch_initialized: bool = False
    device: torch.device

    def __init__(self, config: Config):
        self.train_loader: DataLoader
        self.test_loader: DataLoader
        self.val_loader: DataLoader | None = None
        self.model: torch.nn.Module

        self.config = config

    @abstractmethod
    def init_dataloaders(self) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def train_model(self) -> None:
        raise NotImplementedError()
    
    @abstractmethod
    def evaluate_model(self) -> float:
        raise NotImplementedError()


    @staticmethod
    def init_torch():
        Experiment.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Using device:", Experiment.device)


    def conduct_full(self):
        if not Experiment.torch_initialized:
            Experiment.init_torch()
            Experiment.torch_initialized = True

        self.seed_environment()
        print(f"Starting experiment '{self.config.name}'. Environment seed: {self.config.seed}.")

        print("  Loading datasets...")
        self.init_dataloaders()
        print("  Training samples:", len(self.train_loader.dataset))
        print("  Test samples:", len(self.test_loader.dataset))
        if self.val_loader is not None:
            print("  Validation samples:", len(self.val_loader.dataset))

        print("  Training model...")
        self.train_model()

        print("  Saving model...")
        self.save_model()
        print("  Model saved successfully!")
        print(f"  Final test accuracy: {self.evaluate_model()}%")


    def seed_environment(self):
        """Reproducibility"""
        seed = self.config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def save_model(self):
        path = self.config.model_weights_filepath
        path.parent.mkdir(exist_ok=True)
        torch.save(
            self.model.state_dict(),
            path
        )
