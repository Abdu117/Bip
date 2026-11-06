"""
Use case (simple): improve a weak pretrained model with EWC.

Step 1 (main.py): the CNN was pretrained on only a few images per class (e.g. 100).
Step 2 (this experiment): continue training on DIFFERENT images (e.g. 1000 per class),
with the EWC penalty. Evaluation uses test_samples_per_class test images per class
(the same images as in main.py, because the selection depends only on the seed).
ewc_lambda = 0 switches EWC off (plain fine-tuning = baseline).

How EWC works here:
  1. After loading the pretrained model, the weights are copied and the importance (Fisher
     information) of every weight is measured on the images used for pretraining.
  2. While training on the new images, a penalty is added to the loss. It is large when
     important weights move away from their pretrained values:
         loss = cross_entropy + (lambda / 2) * sum(importance * (weight - old_weight)^2)
"""
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from experiments import Experiment
from experiments.mnist_pretrained import MNISTPretrained, MNISTPretrainedConfig


@dataclass
class MNISTEWCConfig(MNISTPretrainedConfig):
    pretrained_weights_filepath: Path   # weights saved by main.py
    ewc_lambda: float                   # EWC strength, 0 = no EWC
    new_samples_per_class: int          # images per class used in step 2 (e.g. 1000)
    fisher_samples: int = 2000          # max. images used to estimate the importance of the weights
                                        # (if the pretraining set is smaller, all of it is used)


class MNISTEWC(MNISTPretrained):
    def __init__(self, config: MNISTEWCConfig):
        super().__init__(config)
        self.config = config
        self.results: dict[str, float] = {}   # filled in train_model(), read by main_usecase2.py

    # ---------------------------------------------------------------- data
    def init_dataloaders(self):
        full_train, test = MNISTPretrained._load_MNIST()
        test = self._select_test_images(test)   # evaluate only on test_samples_per_class per digit

        # Pretraining images = samples_per_class images per digit. The selection depends only on
        # seed and samples_per_class, so with the same values as in main.py we get exactly the
        # images the model was pretrained on.
        old_data = self._halve_MNIST(full_train)
        # New training images = new_samples_per_class images per digit that were NOT used for
        # pretraining.
        new_data = self._pick_new_images(full_train, set(old_data.indices))

        bs = self.config.batch_size
        # Pretraining images: only used to measure the importance of the weights (Fisher)
        self.old_loader = DataLoader(old_data, batch_size=bs, shuffle=True)
        # New images: the data this step is trained on
        self.train_loader = DataLoader(new_data, batch_size=bs, shuffle=True)
        # Test images (balanced, test_samples_per_class per digit), only for evaluation
        self.test_loader = DataLoader(test, batch_size=bs, shuffle=False)

    def _pick_new_images(self, full_train, used: set) -> Subset:
        """new_samples_per_class random images of every digit, none of them in `used`."""
        targets = full_train.targets
        chosen = []
        for digit in range(10):
            candidates = [i for i in torch.where(targets == digit)[0].tolist() if i not in used]
            generator = torch.Generator().manual_seed(self.config.seed + 100 + digit)
            order = torch.randperm(len(candidates), generator=generator)
            chosen += [candidates[j] for j in order[:self.config.new_samples_per_class].tolist()]
        return Subset(full_train, chosen)

    # ------------------------------------------------------------ training
    def train_model(self):
        # New CNN + loss + optimizer, then load the pretrained weights into the CNN.
        self.model, criterion, optimizer = self._init_training_objects()
        state = torch.load(self.config.pretrained_weights_filepath, map_location=Experiment.device)
        self.model.load_state_dict(state)
        pretrained_accuracy = self.evaluate_model()
        print(f"    Pretrained model | Test accuracy: {pretrained_accuracy:.2f}%")

        # Remember the old weights and how important each weight is (skipped for lambda = 0).
        if self.config.ewc_lambda > 0:
            self._compute_fisher()

        for epoch in range(self.config.epochs):
            loss, train_accuracy = self._train_one_epoch(criterion, optimizer)
            test_accuracy = self.evaluate_model()
            print(
                f"    Epoch [{epoch + 1}/{self.config.epochs}] "
                f"Loss: {loss:.4f} | Train Accuracy: {train_accuracy:.2f}% | "
                f"Test Accuracy: {test_accuracy:.2f}%"
            )

        # Store the accuracies so that main_usecase2.py can save them (only evaluation,
        # the training itself is unchanged).
        self.results = {
            "pretrained_accuracy": pretrained_accuracy,
            "final_accuracy": test_accuracy if self.config.epochs > 0 else pretrained_accuracy,
        }

    def _train_one_epoch(self, criterion, optimizer):
        """Same as in MNISTPretrained, plus the EWC penalty."""
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in self.train_loader:
            images = images.to(Experiment.device)
            labels = labels.to(Experiment.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            if self.config.ewc_lambda > 0:
                loss = loss + (self.config.ewc_lambda / 2) * self._ewc_penalty()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(self.train_loader), 100 * correct / total

    # ----------------------------------------------------------------- EWC
    def _compute_fisher(self):
        """Save the current weights and measure how important each weight is."""
        params = list(self.model.named_parameters())
        self.old_params = {name: p.detach().clone() for name, p in params}
        self.fisher = {name: torch.zeros_like(p) for name, p in params}

        self.model.eval()
        count = 0
        for image, _ in DataLoader(self.old_loader.dataset, batch_size=1, shuffle=True):
            self.model.zero_grad()
            log_probs = F.log_softmax(self.model(image.to(Experiment.device)), dim=1)
            label = torch.multinomial(log_probs.exp(), 1).item()   # label sampled from the model
            (-log_probs[0, label]).backward()
            for name, p in params:
                self.fisher[name] += p.grad ** 2                   # squared gradient = importance
            count += 1
            if count >= self.config.fisher_samples:
                break

        for name in self.fisher:
            self.fisher[name] /= count
        self.model.zero_grad()

    def _ewc_penalty(self):
        """Large if important weights moved away from their pretrained values."""
        return sum(
            (self.fisher[name] * (p - self.old_params[name]) ** 2).sum()
            for name, p in self.model.named_parameters()
        )