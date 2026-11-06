"""
main_usecase1_search.py
=======================
Use case 1: automatic search over several combinations of epochs, learning rate and lambda.

Every combination is one normal run of use case 1 (exactly the same procedure as
main_usecase1.py: step 1 on the old classes, then plain fine-tuning and EWC on the new class).
At the end, all combinations are shown in a table, sorted from best to worst, and the best
combination is printed. All results are also saved to
.out/usecase1_search_old<classes>_new<classes>_<date>_<time>.csv.

"Best" = highest SCORE of the EWC model, computed from the accuracy on the old classes and on
the new classes (see SCORE below).

Run from the project root:   python main_usecase1_search.py
"""
import csv
import itertools
from datetime import datetime
from pathlib import Path

from experiments import MNISTClassIncremental, MNISTClassIncrementalConfig

# ---------------- fixed settings (same meaning as in main_usecase1.py) ----------------
OLD_CLASSES = [4,6]           # digits learned in step 1
NEW_CLASSES = [1]              # digits added in step 2
SAMPLES_PER_OLD_CLASS = None   # images per old digit (None = all available)
SAMPLES_PER_NEW_CLASS = None   # images per new digit (None = all available)
BATCH = 64                     # images per weight update
SEED = 42                      # the same seed for every combination -> fair comparison

# ---------------- values to try: every combination is run ----------------
EPOCHS_VALUES = [5, 8, 10]                  # epochs in step 1 and step 2
LR_VALUES = [0.0001, 0.00005]               # learning rates of Adam
LAMBDA_VALUES = [500_000, 1_000_000, 2_000_000]   # EWC strengths
# -> 3 * 2 * 3 = 18 runs. Every extra value multiplies the run time.

# ---------------- how "best" is measured ----------------
# "mean": average of old-class and new-class accuracy (both count equally)
# "min" : the worse of the two (prefers combinations where BOTH are good)
SCORE = "mean"
# ------------------------------------------------------------------------------------


def score(old: float, new: float) -> float:
    return (old + new) / 2 if SCORE == "mean" else min(old, new)


def main():
    combinations = list(itertools.product(EPOCHS_VALUES, LR_VALUES, LAMBDA_VALUES))
    rows = []

    for i, (epochs, lr, ewc_lambda) in enumerate(combinations, start=1):
        print(f"\n######## Run {i}/{len(combinations)}: epochs={epochs}, lr={lr}, lambda={ewc_lambda:g} ########")
        config = MNISTClassIncrementalConfig(
            name=f"Usecase 1 search - epochs={epochs}, lr={lr}, lambda={ewc_lambda:g}",
            seed=SEED,
            model_weights_filepath=Path(f".out/usecase1_e{epochs}_lr{lr}_lam{ewc_lambda:g}.pth"),
            old_classes=OLD_CLASSES,
            new_classes=NEW_CLASSES,
            samples_per_old_class=SAMPLES_PER_OLD_CLASS,
            samples_per_new_class=SAMPLES_PER_NEW_CLASS,
            ewc_lambda=ewc_lambda,
            epochs=epochs,
            learning_rate=lr,
            batch_size=BATCH,
        )
        experiment = MNISTClassIncremental(config)
        experiment.conduct_full()

        r = experiment.results
        rows.append({
            "pretrained_classes": " ".join(str(c) for c in OLD_CLASSES),
            "added_classes": " ".join(str(c) for c in NEW_CLASSES),
            "epochs": epochs,
            "lr": lr,
            "lambda": ewc_lambda,
            "ewc_old": r["ewc_old"],
            "ewc_new": r["ewc_new"],
            "score": score(r["ewc_old"], r["ewc_new"]),
        })

    # ---- table, sorted from best to worst ----
    rows.sort(key=lambda row: row["score"], reverse=True)
    print(f"\n======== Results (EWC model, sorted by score = {SCORE} of pretrained and added) ========")
    print(f"Pretrained classes: {OLD_CLASSES} | Added classes: {NEW_CLASSES}")
    print(f"{'epochs':>6} {'lr':>9} {'lambda':>10} | {'pretrained':>10} {'added':>8} {'score':>7}")
    for row in rows:
        print(
            f"{row['epochs']:>6} {row['lr']:>9g} {row['lambda']:>10g} | "
            f"{row['ewc_old']:>9.2f}% {row['ewc_new']:>7.2f}% {row['score']:>6.2f}%"
        )

    best = rows[0]
    print(
        f"\nBEST combination: epochs={best['epochs']}, lr={best['lr']:g}, lambda={best['lambda']:g}\n"
        f"  pretrained classes {OLD_CLASSES}: {best['ewc_old']:.2f}% | added classes {NEW_CLASSES}: {best['ewc_new']:.2f}% "
        f"| score: {best['score']:.2f}%"
    )

    # ---- save everything as CSV (open it with Excel, or use it for the report) ----
    # The file name contains the old and new classes plus date and time, e.g.
    # usecase1_search_old0-3_new8_2026-11-06_14-02-31.csv, so older results are kept.
    old_tag = "-".join(str(c) for c in OLD_CLASSES)
    new_tag = "-".join(str(c) for c in NEW_CLASSES)
    out = Path(f".out/usecase1_search_old{old_tag}_new{new_tag}_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"All results saved to {out}")


if __name__ == "__main__":
    main()