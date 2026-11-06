"""
main_usecase2.py
================
STEP 2 of the continual-learning use case. Run it AFTER main.py:   python main_usecase2.py

main.py pretrained a weak CNN on 100 images per digit and saved it to
.out/mnist_pretrained_100.pth. This script loads that model and trains it further on
1,000 OTHER images per digit, with EWC (Elastic Weight Consolidation) to limit forgetting.
It repeats this for several EWC strengths (lambda) so that they can be compared.
Evaluation uses the same 200 test images per digit as main.py.

At the end, a table with all results is printed (sorted from best to worst), and all
results are saved to .out/usecase2_results_<date>_<time>.csv.
"""
import csv
from datetime import datetime
from pathlib import Path

from experiments.mnist_ewc_usecase2 import MNISTEWC, MNISTEWCConfig

# ---------------- settings ----------------
# lambda = EWC strength. 0 = plain fine-tuning without EWC (the baseline to compare
# against). Larger lambda = the weights stay closer to the pretrained model.
# Mostly small values: with a large lambda the model is tied to the weak pretrained
# weights and cannot learn enough from the new data.
LAMBDA_VALUES = [0, 0.1, 1, 10, 100]
SEED = 42                     # same seed as in main.py
SAMPLES_PER_CLASS = 100       # same as in main.py: pretraining images per digit (NOT used here)
TEST_SAMPLES_PER_CLASS = 200  # same as in main.py: test images per digit (200 * 10 = 2,000)
NEW_SAMPLES_PER_CLASS = 1000  # new training images per digit (1000 * 10 = 10,000)
EPOCHS = 5                    # passes over the 10,000 new training images
LEARNING_RATE = 0.001         # step size of the Adam optimizer
BATCH_SIZE = 64               # images per optimisation step
# ------------------------------------------


def main():
    rows = []

    for ewc_lambda in LAMBDA_VALUES:
        config = MNISTEWCConfig(
            # Human-readable name, only used for the log output.
            name=f"Continue pretrained model with more data (lambda={ewc_lambda})",
            seed=SEED,
            samples_per_class=SAMPLES_PER_CLASS,
            test_samples_per_class=TEST_SAMPLES_PER_CLASS,
            new_samples_per_class=NEW_SAMPLES_PER_CLASS,
            # Where the model after this step is saved (one file per lambda).
            model_weights_filepath=Path(f".out/mnist_ewc_lambda_{ewc_lambda}.pth"),
            # The weak model from main.py, used as the starting point.
            pretrained_weights_filepath=Path(".out/mnist_pretrained_100.pth"),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            ewc_lambda=ewc_lambda,
        )
        # conduct_full() runs: load data -> train (with EWC) -> save -> evaluate.
        experiment = MNISTEWC(config)
        experiment.conduct_full()
        print()

        r = experiment.results
        rows.append({
            "pretrain_images_per_class": SAMPLES_PER_CLASS,
            "new_images_per_class": NEW_SAMPLES_PER_CLASS,
            "test_images_per_class": TEST_SAMPLES_PER_CLASS,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "lambda": ewc_lambda,
            "pretrained_accuracy": r["pretrained_accuracy"],
            "final_accuracy": r["final_accuracy"],
            "gain": r["final_accuracy"] - r["pretrained_accuracy"],
        })

    # ---- table, sorted from best to worst final accuracy ----
    rows.sort(key=lambda row: row["final_accuracy"], reverse=True)
    print("======== Results (sorted by final test accuracy) ========")
    print(f"Pretrained model: {rows[0]['pretrained_accuracy']:.2f}% | epochs={EPOCHS}, lr={LEARNING_RATE}")
    print(f"{'lambda':>8} | {'final':>8} {'gain':>9}")
    for row in rows:
        print(f"{row['lambda']:>8g} | {row['final_accuracy']:>7.2f}% {row['gain']:>+8.2f}")

    best = rows[0]
    print(
        f"\nBEST lambda: {best['lambda']:g} -> {best['final_accuracy']:.2f}% "
        f"({best['gain']:+.2f} percentage points compared to the pretrained model)"
    )

    # ---- save everything as CSV (a new file for every run, so older results are kept) ----
    out = Path(f".out/usecase2_results_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"All results saved to {out}")


# Run main() only when this file is executed directly, not when it is imported.
if __name__ == "__main__":
    main()