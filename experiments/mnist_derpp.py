from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from experiments import Experiment, MnistDer, MnistDerConfig


@dataclass
class MnistDerppConfig(MnistDerConfig):
    beta: float


class MnistDerpp(MnistDer):
    def __init__(self, config: MnistDerppConfig):
        # For type checking and Experiment.from_kwargs
        super().__init__(config)
        self.config = config


    def _train_on_batch(
        self,
        criterion: nn.modules.loss._Loss,
        optimizer: optim.Optimizer,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[float, dict[str, float]]:
        """Run one optimisation step and return loss and accuracy."""
        self.model.train()

        images = images.to(Experiment.device)
        labels = labels.to(Experiment.device)

        optimizer.zero_grad()

        outputs = self.model(images)
        loss = criterion(outputs, labels)

        replay = self.model.sample_buffer(images.size(0))
        if replay is not None:
            replay_images, replay_labels, replay_logits = replay
            current_replay_logits = self.model(replay_images)

            # Dark knowledge: match the logits the model produced when the
            # buffered samples were first seen
            replay_mse = F.mse_loss(current_replay_logits, replay_logits)

            # Replay labels: keep learning the true classes of buffered samples
            replay_ce = criterion(current_replay_logits, replay_labels)

            loss = (
                loss
                + self.config.alpha * replay_mse
                + self.config.beta * replay_ce
            )

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
