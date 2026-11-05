"""
main_usecase1.py
================
Use case 1 (framework version): train on a few classes, then ADD a new class with EWC.
Same procedure and same settings as usecase1.py, only run through the framework.

Run from the project root:   python main_usecase1.py
"""
from pathlib import Path

from experiments import MNISTClassIncremental, MNISTClassIncrementalConfig

# ---------------- settings: change them here, nothing else is hard-coded ----------------
OLD_CLASSES = [0, 3]     # digits learned in step 1 (any digits 0-9, any number of them)
NEW_CLASSES = [8]        # digits added in step 2 (must be different from OLD_CLASSES)

SAMPLES_PER_OLD_CLASS = None   # images PER old digit used in step 1.
                               #   None = use all that are available (about 2700-3100 per digit)
                               #   e.g. 500 = only 500 images of each old digit
SAMPLES_PER_NEW_CLASS = None   # images PER new digit used in step 2 (same rules as above)

EWC_LAMBDA = 1000000     # EWC strength: higher = old classes better protected, but new
                         # classes are learned worse. 0 would mean no EWC at all.
EPOCHS = 8               # passes over the training data (used in step 1 AND step 2)
LR = 0.0001              # learning rate of the Adam optimizer (step size of the weight updates)
BATCH = 64               # images per weight update
SEED = 42                # seed of the random number generator: same seed = same data split,
                         # same start weights, same batch order (change it to test robustness)
# -----------------------------------------------------------------------------------------


def main():
    config = MNISTClassIncrementalConfig(
        name=f"Usecase 1 - classes {OLD_CLASSES} + {NEW_CLASSES}",
        seed=SEED,
        model_weights_filepath=Path(".out/usecase1_ewc.pth"),   # the EWC model is saved here
        old_classes=OLD_CLASSES,
        new_classes=NEW_CLASSES,
        samples_per_old_class=SAMPLES_PER_OLD_CLASS,
        samples_per_new_class=SAMPLES_PER_NEW_CLASS,
        ewc_lambda=EWC_LAMBDA,
        epochs=EPOCHS,
        learning_rate=LR,
        batch_size=BATCH,
    )
    # conduct_full() runs: seed -> load data -> step 1, plain fine-tuning, EWC -> save -> evaluate
    MNISTClassIncremental(config).conduct_full()


if __name__ == "__main__":
    main()
