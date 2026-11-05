from abc import abstractmethod
from dataclasses import asdict, dataclass, fields, is_dataclass
import hashlib
import json
import numpy as np
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader


RESULTS_DIR = Path(".results")


@dataclass
class Config:
    name: str
    seed: int


@dataclass
class Measures:
    """Container for logged measures. Every field must be a list."""


class Experiment:
    torch_initialized: bool = False
    device: torch.device
    measures_type: type[Measures] = Measures

    def __init__(self, config: Config):
        """Store the experiment configuration and initialise its measures container."""
        self.train_loader: DataLoader
        self.test_loader: DataLoader
        self.val_loader: DataLoader | None = None
        self.model: torch.nn.Module

        self.config = config
        self.measures = self.measures_type()

    @abstractmethod
    def init_dataloaders(self) -> None:
        """Create the training and test dataloaders for the experiment."""
        raise NotImplementedError()
    
    @abstractmethod
    def train_model(self) -> None:
        """Train the experiment's model and log its measures."""
        raise NotImplementedError()
    
    @abstractmethod
    def evaluate_model(self) -> float:
        """Return the model's accuracy on the test set as a percentage."""
        raise NotImplementedError()

    @abstractmethod
    def init_model(self) -> torch.nn.Module:
        """Create and return the model used by the experiment."""
        raise NotImplementedError()


    @property
    def result_dir(self) -> Path:
        """Directory where this experiment's cached results are stored."""
        return RESULTS_DIR / self.config.name / self.config_hash()

    @property
    def weights_path(self) -> Path:
        """Path to the cached model weights."""
        return self.result_dir / "weights.pth"

    @property
    def measures_path(self) -> Path:
        """Path to the cached measures file."""
        return self.result_dir / "measures.json"

    @property
    def config_path(self) -> Path:
        """Path to the cached configuration file."""
        return self.result_dir / "config.json"


    def conduct_full(self):
        """Run the experiment, or replay the stored results when cached."""
        if not Experiment.torch_initialized:
            Experiment.init_torch()
            Experiment.torch_initialized = True

        print(f"Starting experiment '{self.config.name}'. Environment seed: {self.config.seed}.")

        # A completed run has its config, weights and measures.
        if self.load_measures():
            print("  Experiment detected in the cache. Printing stored results:")
            self.restore_cached_state()
            self.print_measures()
            return

        self.seed_environment()

        print("  Loading datasets...")
        self.init_dataloaders()
        print(f"  Training samples: {len(self.train_loader.dataset)}")
        print(f"  Test samples: {len(self.test_loader.dataset)}")
        if self.val_loader is not None:
            print(f"  Validation samples: {len(self.val_loader.dataset)}")

        print("  Training model...")
        self.train_model()

        print("  Saving experiment (config, measures, model)...")
        self.save_to_disk()
        print("  Experiment saved successfully!")
        print(f"  Final test accuracy: {self.evaluate_model()}%")

    @staticmethod
    def init_torch():
        """Select the computation device and store it on the Experiment class."""
        Experiment.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Using device:", Experiment.device)


    def seed_environment(self):
        """Seed Python, NumPy and PyTorch for reproducible runs."""
        seed = self.config.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def config_hash(self) -> str:
        """Return a stable hash of the config contents, used to cache results."""
        payload = json.dumps(
            self.config_to_dict(self.config),
            sort_keys=True
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()


    @classmethod
    def config_to_dict(cls, config: Config) -> dict:
        """Convert a config dataclass into a JSON-serialisable dictionary."""
        return {
            field.name: cls._value_to_dict(getattr(config, field.name))
            for field in fields(config)
        }


    @classmethod
    def _value_to_dict(cls, value):
        """Recursively convert a config value into a JSON-serialisable value."""
        if is_dataclass(value):
            return cls.config_to_dict(value)

        # Nested experiments are identified by their result path instead of
        # duplicating their whole config in the parent config.
        if isinstance(value, Experiment):
            return f"{value.config.name}/{value.config_hash()}"

        if isinstance(value, (list, tuple)):
            return [cls._value_to_dict(item) for item in value]

        return value


    def restore_cached_state(self) -> None:
        """Build the model and load the weights cached for this experiment."""
        self.model = self.init_model()
        state = torch.load(self.weights_path, map_location=Experiment.device)
        self.model.load_state_dict(state)
        self.model.eval()


    def log_measure(self, **values: int | float) -> None:
        """Append one value per field to the measures and print the formatted row."""
        for name, value in values.items():
            getattr(self.measures, name).append(value)

        print(self.format_measure(values))


    def format_measure(self, measure: dict) -> str:
        """Return a one-line, human-readable representation of a measure."""
        parts = []
        for name, value in measure.items():
            label = name.replace("_", " ").capitalize()
            if isinstance(value, float):
                value = f"{value:.4f}"
            parts.append(f"{label}: {value}")

        return "    " + " | ".join(parts)


    def load_measures(self) -> bool:
        """Load the cached measures, returning whether they were found and valid."""
        try:
            with self.measures_path.open(encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            return False

        if not isinstance(data, dict):
            return False

        self.measures = self.measures_type()
        for field in fields(self.measures):
            if field.name in data:
                setattr(self.measures, field.name, data[field.name])

        return True


    def print_measures(self) -> None:
        """Print every cached measure using the common formatter."""
        fields_ = fields(self.measures)
        nb_measures = max(
            (len(getattr(self.measures, field.name)) for field in fields_),
            default=0
        )

        for index in range(nb_measures):
            measure = {
                field.name: getattr(self.measures, field.name)[index]
                for field in fields_
                if index < len(getattr(self.measures, field.name))
            }
            print(self.format_measure(measure))


    def save_to_disk(self) -> None:
        """Save the config, measures and model weights to the result directory."""
        # Config
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as file:
            json.dump(self.config_to_dict(self.config), file, indent=4)

        # Measures
        self.measures_path.parent.mkdir(parents=True, exist_ok=True)
        with self.measures_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(self.measures), file, indent=4)

        # Model
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            self.weights_path
        )
