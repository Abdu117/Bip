"""
main.py
=======
Entry point of the project. Run it with:   python main.py   (or: uv run main.py)

It performs STEP 1 of the continual-learning use case: pretrain a CNN on one half of the
MNIST training set (30,000 images) and save the weights to .out/mnist_pretrained.pth.
That checkpoint is later the starting point for the continual-learning step (EWC).
"""
from pathlib import Path
from experiments import MNISTPretrained, MNISTPretrainedConfig


def main():
    # All settings of the run live in one config object, so an experiment is fully
    # described (and reproducible) by this block.
    config = MNISTPretrainedConfig(
        # Human-readable name, only used for the log output.
        name="Usecase 1 - MNIST pretrained",
        # Seed for all random number generators -> same split, same initial weights,
        # same shuffling on every run. IMPORTANT: a later experiment that needs "the
        # other half" of the data must use the same seed AND samples_per_class,
        # otherwise it would select a different half.
        seed=42,
        # Where the trained weights are written. The folder .out is ignored by git.
        model_weights_filepath=Path(".out/mnist_pretrained.pth"),

        # Number of images per optimisation step (the model's weights are updated once
        # per batch of 64 images).
        batch_size=64,
        # Number of complete passes over the 30,000 training images.
        # (Note: README.md still says 10 epochs; the value that is actually used is 5.)
        epochs=5,
        # Step size of the Adam optimizer. Too large -> unstable training, too small ->
        # very slow learning. 0.001 is the usual default for Adam.
        learning_rate=0.001,
        # How many images of EACH digit (0-9) go into the pretraining half.
        # 3000 * 10 classes = 30,000 images = about half of MNIST's 60,000.
        samples_per_class=3000,
    )

    # conduct_full() runs the whole pipeline: load data -> train -> save -> evaluate.
    MNISTPretrained(config).conduct_full()


# Standard Python idiom: run main() only when this file is executed directly,
# not when it is imported from another file.
if __name__ == "__main__":
    main()