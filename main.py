"""
main.py
=======
Entry point of the project. Run it with:   python main.py   (or: uv run main.py)

It performs STEP 1 of the continual-learning use case: pretrain a CNN on only a few
images per class (100 per digit = 1,000 images in total) and save the weights to
.out/mnist_pretrained_100.pth. The model is deliberately weak: it is the starting
point for the next step (main_usecase2.py), where it is trained further on other images
(1,000 per digit) with EWC. Evaluation uses 200 test images per digit (2,000 in total).
"""
from pathlib import Path
from experiments import MNISTPretrained, MNISTPretrainedConfig


def main():
    # All settings of the run live in one config object, so an experiment is fully
    # described (and reproducible) by this block.
    config = MNISTPretrainedConfig(
        # Human-readable name, only used for the log output.
        name="Pretraining on 100 images per class",
        # Seed for all random number generators -> same image selection, same initial
        # weights, same shuffling on every run. IMPORTANT: main_usecase2.py must use the same
        # seed AND the same samples_per_class, otherwise it would not know which images
        # were already used for pretraining.
        seed=42,
        # Where the trained weights are written. main_usecase2.py loads this file.
        # The folder .out is ignored by git.
        model_weights_filepath=Path(".out/mnist_pretrained_100.pth"),

        # Number of images per optimisation step (the model's weights are updated once
        # per batch of 64 images).
        batch_size=64,
        # Number of complete passes over the 1,000 training images
        # (1,000 images / batch size 64 = about 16 weight updates per epoch).
        epochs=5,
        # Step size of the Adam optimizer. Too large -> unstable training, too small ->
        # very slow learning. 0.001 is the usual default for Adam.
        learning_rate=0.001,
        # How many images of EACH digit (0-9) are used for pretraining.
        # 100 * 10 classes = 1,000 images. Must be the same value in main_usecase2.py.
        samples_per_class=100,
        # How many test images of EACH digit are used for evaluation.
        # 200 * 10 classes = 2,000 test images (one image = 0.05 percentage points).
        # Must be the same value in main_usecase2.py.
        test_samples_per_class=200,
    )

    # conduct_full() runs the whole pipeline: load data -> train -> save -> evaluate.
    MNISTPretrained(config).conduct_full()


# Standard Python idiom: run main() only when this file is executed directly,
# not when it is imported from another file.
if __name__ == "__main__":
    main()