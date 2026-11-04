"""
models/cnn.py
=============
The neural network used in every experiment: a small Convolutional Neural Network (CNN)
for MNIST digit classification.

A CNN is a good fit for images because its convolution layers look at small local patches
(here 3x3 pixels) and re-use the same filter at every position of the image. That lets the
network detect things like edges and strokes anywhere in the picture with very few weights.
"""
import torch.nn as nn


class CNN(nn.Module):
    """Small CNN for MNIST: input (N, 1, 28, 28), output (N, 10).

    Notation: N = batch size (how many images are processed at once),
              1 = number of colour channels (grayscale), 28 x 28 = image height x width,
              10 = one score per digit class (0-9).
    """

    def __init__(self):
        # Must be called first: it sets up PyTorch's internal machinery so that layers
        # assigned to `self` below are registered as parameters of this model.
        super().__init__()

        # ---- Part 1: feature extractor (finds patterns in the image) ----
        # Shape changes are written as (channels, height, width) per image.
        self.features = nn.Sequential(
            # 16 filters of size 3x3 slide over the image. padding=1 keeps the size at 28x28.
            # (1, 28, 28) -> (16, 28, 28).   Parameters: 1*16*3*3 + 16 biases = 160
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            # Non-linearity: replaces negative values with 0. Without it, stacking layers
            # would collapse into one big linear function.
            nn.ReLU(),
            # Keeps the maximum of every 2x2 block -> halves height and width.
            # (16, 28, 28) -> (16, 14, 14)
            nn.MaxPool2d(2),
            # 32 filters, each looking at all 16 incoming channels.
            # (16, 14, 14) -> (32, 14, 14).   Parameters: 16*32*3*3 + 32 = 4,640
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # (32, 14, 14) -> (32, 7, 7)
            nn.MaxPool2d(2),
        )

        # ---- Part 2: classifier (turns the found patterns into 10 class scores) ----
        self.classifier = nn.Sequential(
            # Unrolls (32, 7, 7) into one vector of 32*7*7 = 1568 numbers per image.
            nn.Flatten(),
            # Fully connected layer: 1568 -> 128.   Parameters: 1568*128 + 128 = 200,832
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            # Output layer: 128 -> 10, one raw score ("logit") per digit.
            # Parameters: 128*10 + 10 = 1,290
            nn.Linear(128, 10),
        )
        # Total: 160 + 4,640 + 200,832 + 1,290 = 206,922 trainable parameters.
        # Most of them sit in the first Linear layer.

    def forward(self, x):
        # Called automatically when you write `model(images)`.
        # Returns raw scores (logits), NOT probabilities: there is deliberately no softmax
        # here, because nn.CrossEntropyLoss applies log-softmax internally, and for
        # predictions argmax over the raw scores gives the same digit as argmax over
        # probabilities.
        return self.classifier(self.features(x))