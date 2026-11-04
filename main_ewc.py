"""
main_ewc.py
===========
STEP 2 of the continual-learning use case. Run it AFTER main.py:   python main_ewc.py

main.py pretrained a weak CNN on 100 images per digit and saved it to
.out/mnist_pretrained_100.pth. This script loads that model and trains it further on
1,000 OTHER images per digit, with EWC (Elastic Weight Consolidation) to limit forgetting.
It repeats this for several EWC strengths (lambda) so that they can be compared.
Evaluation uses the same 20 test images per digit as main.py.
"""
from pathlib import Path
from experiments.mnist_ewc import MNISTEWC, MNISTEWCConfig


def main():
    # lambda = EWC strength. 0 = plain fine-tuning without EWC (the baseline to compare
    # against). Larger lambda = the weights stay closer to the pretrained model.
    # Each value gets its own run; add or remove values to compare more or fewer settings.
    # Only small values: with a large lambda the model is tied to the weak pretrained
    # weights and cannot learn enough from the new data.
    for ewc_lambda in [0, 0.1, 1, 10 , 100]:
        config = MNISTEWCConfig(
            # Human-readable name, only used for the log output.
            name=f"Continue pretrained model with more data (lambda={ewc_lambda})",
            # Same seed as in main.py: needed to find out which images were already used
            # for pretraining, and to evaluate on the same test images.
            seed=42,
            # Same value as in main.py: pretraining used this many images per digit
            # (100 * 10 digits = 1,000 images). These images are NOT used for training here.
            samples_per_class=100,
            # Same value as in main.py: number of test images per digit (20 * 10 = 200).
            test_samples_per_class=200,
            # Images per digit used for training in this step (1000 * 10 = 10,000 images).
            # They are different from the images used in main.py.
            new_samples_per_class=1000,
            # Where the model after this step is saved (one file per lambda).
            model_weights_filepath=Path(f".out/mnist_ewc_lambda_{ewc_lambda}.pth"),
            # The weak model from main.py, used as the starting point.
            pretrained_weights_filepath=Path(".out/mnist_pretrained_100.pth"),

            # Number of images per optimisation step.
            batch_size=64,
            # Number of complete passes over the 10,000 new training images.
            # More epochs give the model more time to learn from the new data.
            epochs=5,
            # Step size of the Adam optimizer.
            learning_rate=0.001,
            # EWC strength for this run (see the loop above).
            ewc_lambda=ewc_lambda,
        )
        # conduct_full() runs: load data -> train (with EWC) -> save -> evaluate.
        MNISTEWC(config).conduct_full()
        print()


# Run main() only when this file is executed directly, not when it is imported.
if __name__ == "__main__":
    main()