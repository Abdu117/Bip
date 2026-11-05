"""
experiments/mnist_class_incremental.py
======================================
Use case 1 in the framework: train on a few classes, then ADD a new class with EWC.

This is exactly the same procedure as usecase1.py, only split into the framework:
  1. Split the 60,000 training images randomly into two halves.
  2. Step 1: train a CNN on the OLD classes from the first half (only once).
  3. Remember the old weights and how important each weight is (Fisher information).
  4. Step 2, twice from the SAME step-1 model, on the NEW classes from the second half:
       a) plain fine-tuning (no EWC)   -> baseline
       b) with the EWC penalty
  5. Print the accuracy on the old and on the new classes for both.

All settings come from the config (main_usecase1.py), nothing is hard-coded.
With the same settings, the random numbers are used in the same order as in usecase1.py.
"""
import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from experiments import Experiment, Config
from experiments.mnist_pretrained import MNISTPretrained
from models import CNN


@dataclass
class MNISTClassIncrementalConfig(Config):
    old_classes: list[int]              # digits learned in step 1 (any digits 0-9)
    new_classes: list[int]              # digits added in step 2 (different from old_classes)
    samples_per_old_class: int | None   # images per old digit in step 1 (None = all available)
    samples_per_new_class: int | None   # images per new digit in step 2 (None = all available)
    ewc_lambda: float                   # EWC strength: higher = old classes better protected
    epochs: int                         # epochs in step 1 AND in step 2
    learning_rate: float                # learning rate of Adam
    batch_size: int                     # images per weight update
    fisher_samples: int = 2000          # images used to estimate the importance of the weights

    def __post_init__(self):
        assert not set(self.old_classes) & set(self.new_classes), "old and new classes must be different"


class MNISTClassIncremental(Experiment):
    def __init__(self, config: MNISTClassIncrementalConfig):
        super().__init__(config)
        self.config = config

    # ---------------------------------------------------------------- data
    @staticmethod
    def _only_classes(ds, indices, classes, n_per_class=None) -> Subset:
        """Subset of ds: from the given indices keep only images whose label is in `classes`,
        at most `n_per_class` images per class (None = no limit).
        The indices are in random order, so the first n images of a class are a random choice."""
        labels = ds.targets.tolist()
        taken = {c: 0 for c in classes}        # how many images of each class were picked so far
        chosen = []
        for i in indices:
            c = labels[i]
            if c in taken and (n_per_class is None or taken[c] < n_per_class):
                chosen.append(i)
                taken[c] += 1
        return Subset(ds, chosen)

    def init_dataloaders(self):
        cfg = self.config
        train_ds, self.test_ds = MNISTPretrained._load_MNIST()

        # Split the training images randomly into two halves (same as usecase1.py).
        perm = torch.randperm(len(train_ds)).tolist()
        half_a, half_b = perm[:30000], perm[30000:]     # half_a for step 1, half_b for step 2

        self.data_a = self._only_classes(train_ds, half_a, cfg.old_classes, cfg.samples_per_old_class)
        self.data_b = self._only_classes(train_ds, half_b, cfg.new_classes, cfg.samples_per_new_class)

        # Loaders the framework expects (used for its log output and final evaluation)
        self.train_loader = DataLoader(self.data_a, batch_size=cfg.batch_size, shuffle=True)
        test_all = self._only_classes(self.test_ds, range(len(self.test_ds)), cfg.old_classes + cfg.new_classes)
        self.test_loader = DataLoader(test_all, batch_size=512)

    # ------------------------------------------------------------ helpers
    def _train(self, model, data, ewc=None, lam=0.0):
        """Train `model` on `data`. If `ewc` is given, the EWC penalty (strength `lam`) is added."""
        loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        model.train()
        for _ in range(self.config.epochs):
            for x, y in loader:
                x, y = x.to(Experiment.device), y.to(Experiment.device)
                loss = F.cross_entropy(model(x), y)
                if ewc is not None:
                    loss = loss + (lam / 2) * self._ewc_penalty(model, ewc)   # EWC: stay close to old weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def _accuracy(self, model, classes):
        """Test accuracy (in %) on the test images of the given classes."""
        model.eval()
        loader = DataLoader(self._only_classes(self.test_ds, range(len(self.test_ds)), classes), batch_size=512)
        correct, total = 0, 0
        for x, y in loader:
            correct += (model(x.to(Experiment.device)).argmax(dim=1).cpu() == y).sum().item()
            total += len(y)
        return 100 * correct / total

    def _compute_ewc(self, model, data):
        """Remember the old weights and how important each weight is (Fisher information)."""
        fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
        model.eval()
        count = 0
        for x, _ in DataLoader(data, batch_size=1, shuffle=True):
            model.zero_grad()
            log_probs = F.log_softmax(model(x.to(Experiment.device)), dim=1)
            label = torch.multinomial(log_probs.exp(), 1).item()     # label sampled from the model
            (-log_probs[0, label]).backward()
            for name, p in model.named_parameters():
                fisher[name] += p.grad ** 2                          # squared gradient = importance
            count += 1
            if count >= self.config.fisher_samples:
                break
        model.zero_grad()
        return {
            "fisher": {name: f / count for name, f in fisher.items()},
            "old": {name: p.detach().clone() for name, p in model.named_parameters()},
        }

    @staticmethod
    def _ewc_penalty(model, ewc):
        """Sum of importance * (change of weight)^2 : big for important weights that moved."""
        return sum(
            (ewc["fisher"][name] * (p - ewc["old"][name]) ** 2).sum()
            for name, p in model.named_parameters()
        )

    def _report(self, title, model):
        cfg = self.config
        print(
            f"{title:<28} old classes {cfg.old_classes}: {self._accuracy(model, cfg.old_classes):6.2f}% | "
            f"new classes {cfg.new_classes}: {self._accuracy(model, cfg.new_classes):6.2f}%"
        )

    # ------------------------------------------------------------ training
    def train_model(self):
        cfg = self.config

        print(f"Step 1: train on classes {cfg.old_classes} ({len(self.data_a)} images)")
        base = CNN().to(Experiment.device)
        self._train(base, self.data_a)
        self._report("After step 1", base)

        ewc = self._compute_ewc(base, self.data_a)

        print(f"\nStep 2: add classes {cfg.new_classes} ({len(self.data_b)} images)")

        plain = copy.deepcopy(base)                  # same starting point for a fair comparison
        self._train(plain, self.data_b)
        self._report("Plain fine-tuning (no EWC)", plain)

        with_ewc = copy.deepcopy(base)
        self._train(with_ewc, self.data_b, ewc=ewc, lam=cfg.ewc_lambda)
        self._report(f"With EWC (lambda={cfg.ewc_lambda})", with_ewc)

        # The EWC model is the one the framework saves and evaluates at the end.
        self.model = with_ewc

    def evaluate_model(self) -> float:
        """Accuracy of the EWC model over old + new classes (the 'Final test accuracy' line)."""
        return self._accuracy(self.model, self.config.old_classes + self.config.new_classes)
