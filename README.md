# MNIST Pretrained CNN

## Overview

A CNN model pretrained on 50% of the MNIST training dataset for the Continual Learning project.

## Dataset

* Dataset: MNIST
* Training samples: 30,000
* Samples per class: 3,000
* Number of classes: 10
* Test samples: 10,000

## Training Configuration

* Architecture: Custom CNN
* Epochs: 10
* Optimizer: Adam
* Learning rate: 0.001
* Batch size: 64
* Loss: Cross-Entropy

## Files

* `mnist_pretrained.pth`: Trained model weights.
* `cnn.py`: CNN architecture.
* `config.json`: Training configuration.
* `pretrain_mnist.py`: Training script.

## Usage

Load the model architecture and checkpoint, then evaluate or continue training.

The MNIST dataset is downloaded automatically using torchvision.
