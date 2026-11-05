from abc import abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
import hashlib
import json
import numpy as np
from pathlib import Path
import random
from typing import get_type_hints, Self

import torch
from torch.utils.data import DataLoader


RESULTS_DIR = Path(".results")


@dataclass
class Config:
    name: str
    group: str
    seed: int


class Experiment:
    torch_initialized: bool = False
    device: torch.device

    def __init__(self, config: Config):
        """Store the experiment configuration and initialise its measures container."""
        self.train_loader: DataLoader
        self.test_loader: DataLoader
        self.val_loader: DataLoader | None = None
        self.model: torch.nn.Module

        self.config = config
        self.measures: dict[str, list] = {}
        self._result_dir: Path | None = None # Needs to be stored since it contains a timestamp


    @classmethod
    def from_kwargs(cls, **kwargs) -> Self:
        """Create an experiment from its config constructor arguments."""
        config_type = get_type_hints(cls.__init__).get("config")
        return cls(config_type(**kwargs))


    @abstractmethod
    def init_dataloaders(self) -> None:
        """Create the training and test dataloaders for the experiment."""
        raise NotImplementedError()
    
    @abstractmethod
    def train_model(self) -> None:
        """Train the experiment's model and log its measures."""
        raise NotImplementedError()
    
    @abstractmethod
    def evaluate_model(self) -> float | dict[str, float]:
        """Return the model's accuracy, overall or per class, as a percentage."""
        raise NotImplementedError()

    @abstractmethod
    def init_model(self) -> torch.nn.Module:
        """Create and return the model used by the experiment."""
        raise NotImplementedError()


    @property
    def result_parent(self) -> Path:
        """Directory containing the cached runs of this experiment."""
        return RESULTS_DIR / self.config.group / self.config.name

    @property
    def result_dir(self) -> Path:
        """Directory where the current run is stored."""
        if self._result_dir is None:
            self._result_dir = self.new_result_dir()
        return self._result_dir

    @property
    def weights_path(self) -> Path:
        """Path to the model weights."""
        return self.result_dir / "weights.pth"

    @property
    def measures_path(self) -> Path:
        """Path to the measures file."""
        return self.result_dir / "measures.json"

    @property
    def config_path(self) -> Path:
        """Path to the configuration file."""
        return self.result_dir / "config.json"

    def conduct_full(self):
        """Run the experiment, or replay the stored results when cached."""
        if not Experiment.torch_initialized:
            Experiment.init_torch()
            Experiment.torch_initialized = True

        print(f"Starting experiment '{self.config.name}'. Environment seed: {self.config.seed}.")

        # A completed run has a directory tagged with the config hash.
        cached_dir = self.find_cached_result_dir()
        if cached_dir is not None:
            self._result_dir = self.refresh_cached_result_dir(cached_dir)

            if self.load_measures():
                print("  Experiment detected in the cache. Printing stored results:")
                self.load_model_from_cache()
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

        final_accuracy = self.evaluate_model()
        if isinstance(final_accuracy, dict):
            print(f"  Final test accuracy: {self.format_value(final_accuracy)}")
        else:
            print(f"  Final test accuracy: {final_accuracy}%")

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
            return f"{value.config.group}/{value.config.name}"

        if isinstance(value, (list, tuple)):
            return [cls._value_to_dict(item) for item in value]

        return value


    def config_hash(self) -> str:
        """Return a short stable hash of the config contents, used to tag results."""
        payload = json.dumps(
            self.config_to_dict(self.config),
            sort_keys=True
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:10]


    def new_result_dir(self) -> Path:
        """Return a fresh result directory named with the current date and config hash."""
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        return self.result_parent / f"{timestamp}-{self.config_hash()}"


    def find_cached_result_dir(self) -> Path | None:
        """Return the newest cached directory whose hash matches this config, if any."""
        if not self.result_parent.is_dir():
            return None

        config_hash = self.config_hash()
        matches = [
            path
            for path in self.result_parent.iterdir()
            if path.is_dir() and path.name.rsplit("-", 1)[-1] == config_hash
        ]
        if not matches:
            return None

        return max(matches)


    def refresh_cached_result_dir(self, cached_dir: Path) -> Path:
        """Rename a cached result directory to the current date and return it."""
        result_dir = self.new_result_dir()
        cached_dir.rename(result_dir)
        return result_dir


    def load_model_from_cache(self) -> None:
        """Build the model and load the weights cached for this experiment."""
        self.model = self.init_model()
        state = torch.load(self.weights_path, map_location=Experiment.device)
        self.model.load_state_dict(state)
        self.model.eval()


    def log_measure(self, **values: int | float | dict) -> None:
        """Append one value per measure and print the formatted row."""
        for name, value in values.items():
            self.measures.setdefault(name, []).append(value)

        print(self.format_measure(values))


    @staticmethod
    def format_value(value) -> str:
        """Return a human-readable representation of a measure value."""
        if isinstance(value, dict):
            return "{" + ", ".join(
                f"{key}: {Experiment.format_value(item)}"
                for key, item in value.items()
            ) + "}"

        if isinstance(value, float):
            return f"{value:.4f}"

        return str(value)


    def format_measure(self, measure: dict) -> str:
        """Return a one-line, human-readable representation of a measure."""
        parts = []
        for name, value in measure.items():
            label = name.replace("_", " ").capitalize()
            parts.append(f"{label}: {self.format_value(value)}")

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

        self.measures = data
        return True


    def print_measures(self) -> None:
        """Print every cached measure using the common formatter."""
        nb_measures = max(
            (len(values) for values in self.measures.values()),
            default=0
        )

        for index in range(nb_measures):
            measure = {
                name: values[index]
                for name, values in self.measures.items()
                if index < len(values)
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
            json.dump(self.measures, file, indent=4)

        # Model
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            self.weights_path
        )
