import torch
import torch.nn as nn


class Cnn(nn.Module):
    """Small CNN for MNIST: input (N, 1, 28, 28), output (N, nb_labels)."""

    def __init__(self, nb_labels: int):
        """Build the convolutional feature extractor and the classification head."""
        super().__init__()
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
            nn.Linear(128, nb_labels),
        )


    @staticmethod
    def from_cnn(base: "Cnn", new_nb_labels: int) -> "Cnn":
        """Return a copy of ``base`` with ``new_nb_labels`` output classes.

        Shared class weights are copied over (truncated when classes are
        removed). Weights for added classes keep the default initialisation of
        a fresh ``CNN``. No training is performed.
        """
        HEAD_INDEX = 3

        old_head = base.classifier[HEAD_INDEX]

        model = Cnn(new_nb_labels).to(
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

            # Copy the class weights and biases that both models share.
            nb_shared = min(old_head.out_features, new_nb_labels)
            model.classifier[HEAD_INDEX].weight[:nb_shared] = old_head.weight[:nb_shared]
            if old_head.bias is not None:
                model.classifier[HEAD_INDEX].bias[:nb_shared] = old_head.bias[:nb_shared]

        return model


    def forward(self, x):
        """Run a batch of images through the network."""
        return self.classifier(self.features(x))
