import torch
import torch.nn as nn


class MnistCnn(nn.Module):
    """Small CNN for MNIST: input (N, 1, 28, 28), output (N, nb_labels)."""

    def __init__(self, classes: list[int]):
        """Build the convolutional feature extractor and the classification head."""
        super().__init__()
        self.classes = classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, len(classes)),
        )


    @staticmethod
    def from_cnn(base: "MnistCnn", new_classes: list[int]) -> "MnistCnn":
        """Return a copy of ``base`` with ``new_classes`` as output classes.

        Shared class weights are copied by class identity, so classes can be
        reordered, removed or added. Weights for added classes keep the default
        initialisation of a fresh ``MnistCnn``. No training is performed.
        """
        HEAD_INDEX = 3

        old_head = base.classifier[HEAD_INDEX]

        model = MnistCnn(new_classes).to(
            device=old_head.weight.device,
            dtype=old_head.weight.dtype,
        )
        head_prefix = f"classifier.{HEAD_INDEX}."

        with torch.no_grad():
            # Copy every layer except the output layer.
            old_state = base.state_dict()
            for key, value in model.state_dict().items():
                if not key.startswith(head_prefix):
                    value.copy_(old_state[key])

            head = model.classifier[HEAD_INDEX]
            old_indices = {
                label: index
                for index, label in enumerate(base.classes)
            }

            # Copy each class by identity, leaving truly new classes with
            # their fresh initialisation.
            for new_index, label in enumerate(new_classes):
                old_index = old_indices.get(label)
                if old_index is None:
                    continue

                head.weight[new_index].copy_(old_head.weight[old_index])
                if old_head.bias is not None:
                    head.bias[new_index].copy_(old_head.bias[old_index])

        return model


    def forward(self, x):
        return self.classifier(self.features(x))
