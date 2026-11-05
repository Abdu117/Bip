from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from experiments import Experiment, MnistUsecase, MnistConfig
from models import MnistCnn, Der


@dataclass
class MnistDerConfig(MnistConfig):
    buffer_size: int
    alpha: float


class MnistDer(MnistUsecase):
    def __init__(self, config: MnistDerConfig):
        # For type checking
        super().__init__(config)
        self.config = config
        self.model: Der


    def init_model(self) -> nn.Module:
        """Adapts the pretrained backbone to the model class set and wrap it in DER."""
        backbone = MnistCnn.from_cnn(
            self.config.pretrained.model,
            new_classes=self.config.all_classes,
        ).to(Experiment.device)

        return Der(backbone, self.config.buffer_size).to(Experiment.device)


    def train_model(self):
        """Train and log the measures of each batch."""
        self.model, criterion, optimizer = self._init_training_objects()

        initial_accuracy = self.evaluate_model()
        print(f"  Number of batches: {len(self.train_loader)}")
        print(f"  Initial test accuracy: {self.format_value(initial_accuracy)}")

        for batch, (images, labels) in enumerate(self.train_loader, start=1):
            loss, train_accuracy = self._train_on_batch(
                criterion, optimizer, images, labels
            )

            test_accuracy = self.evaluate_model()
            buffer_length = self.model.buffer_length

            self.log_measure(
                batch=batch,
                loss=loss,
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy,
                buffer_length=buffer_length
            )


    def _init_training_objects(self) -> tuple[nn.Module, nn.modules.loss._Loss, optim.Optimizer]:
        """Create the model, loss function and optimizer for training."""
        model = self.init_model()

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate
        )

        return model, criterion, optimizer


    def _train_on_batch(
        self,
        criterion: nn.modules.loss._Loss,
        optimizer: optim.Optimizer,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, dict[str, float]]:
        """Run one optimisation step and return loss and per-class train accuracy."""
        self.model.train()

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

        predictions = outputs.argmax(dim=1)
        train_accuracy = {}
        for index in torch.unique(labels).tolist():
            mask = labels == index
            train_accuracy[str(self.config.all_classes[index])] = (
                100 * (predictions[mask] == labels[mask]).sum().item()
                / mask.sum().item()
            )

        train_accuracy["global"] = (
            100 * (predictions == labels).sum().item() / labels.size(0)
        )

        return loss.item(), train_accuracy
