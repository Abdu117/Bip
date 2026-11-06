## Use case 1: Results

Adding a new class with EWC works, but only in a narrow range of settings, and the best
setting depends on the class pair. The best combination reached a score between **82.3 %**
and **94.3 %**, while plain fine-tuning without EWC forgot the old classes completely (0 %).

### Search space

Every run tested all 18 combinations of:

| Parameter | Values |
| --- | --- |
| Epochs (step 1 and 2) | 5, 8, 10 |
| Learning rate | 0.0001, 0.00005 |
| λ | 500,000, 1,000,000, 2,000,000 |

Score = mean of the accuracy on the old classes and on the new class (EWC model).

### Best combination per run

| Run | Old classes | New class | Epochs | Learning rate | λ | Old classes (%) | New class (%) | Score (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D | 4, 6 | 1 | 8 | 0.00005 | 500,000 | 93.87 | 94.80 | 94.33 |
| B | 0, 3 | 8 | 5 | 0.0001 | 1,000,000 | 86.03 | 92.97 | 89.50 |
| A | 0, 3 | 8 | 10 | 0.0001 | 1,000,000 | 84.67 | 92.40 | 88.54 |
| C | 1, 8 | 4 | 10 | 0.00005 | 500,000 | 67.47 | 97.05 | 82.26 |

Only **11 of 72** combinations reached at least 85 % on both the old and the new classes.
Plain fine-tuning (no EWC) reached 0 % on the old classes and 100 % on the new class in every
combination.

### Top 3 combinations per run

| Run | Rank | Epochs | Learning rate | λ | Old classes (%) | New class (%) | Score (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A (0, 3 → 8) | 1 | 10 | 0.0001 | 1,000,000 | 84.67 | 92.40 | 88.54 |
| | 2 | 8 | 0.0001 | 1,000,000 | 88.14 | 87.68 | 87.91 |
| | 3 | 10 | 0.00005 | 500,000 | 89.30 | 86.14 | 87.72 |
| B (0, 3 → 8) | 1 | 5 | 0.0001 | 1,000,000 | 86.03 | 92.97 | 89.50 |
| | 2 | 5 | 0.0001 | 500,000 | 78.98 | 96.95 | 87.96 |
| | 3 | 5 | 0.0001 | 2,000,000 | 89.86 | 85.23 | 87.55 |
| C (1, 8 → 4) | 1 | 10 | 0.00005 | 500,000 | 67.47 | 97.05 | 82.26 |
| | 2 | 10 | 0.00005 | 1,000,000 | 75.82 | 87.88 | 81.85 |
| | 3 | 8 | 0.00005 | 500,000 | 79.14 | 84.01 | 81.57 |
| D (4, 6 → 1) | 1 | 8 | 0.00005 | 500,000 | 93.87 | 94.80 | 94.33 |
| | 2 | 5 | 0.0001 | 1,000,000 | 92.47 | 95.86 | 94.17 |
| | 3 | 10 | 0.00005 | 1,000,000 | 91.03 | 97.27 | 94.15 |

### Where it fails

| Setting | Run A | Run B | Run C | Run D | Effect |
| --- | --- | --- | --- | --- | --- |
| 5 epochs, LR 0.00005: new class (%) | 0 – 23.0 | 0 – 74.8 | 0 – 0.1 | 0 | new class not learned |
| 10 epochs, LR 0.0001: old classes (%) | 76.1 – 90.8 | 58.5 – 61.4 | 0 | 12.8 – 14.0 | old classes forgotten |

Ranges cover the three λ values.

### Observations

1. **Without EWC there is no memory.** Plain fine-tuning always forgets the old classes
   completely, because step 2 only contains images of the new class.
2. **The amount of training (epochs × learning rate) matters most.** Too little and the new
   class is not learned; too much and the old classes are forgotten, even with
   λ = 2,000,000. All best results come from a medium amount of training.
3. **λ shifts the trade-off between old and new.** At a fixed amount of training, a larger λ
   protects the old classes better and lowers the accuracy on the new class. Example run A
   (8 epochs, LR 0.0001): λ = 500,000 gives 79.3 % old / 94.0 % new, λ = 2,000,000 gives
   94.1 % old / 72.4 % new.
4. **The class pair matters.** The same search gives best scores from 94.3 % (4, 6 → 1) down
   to 82.3 % (1, 8 → 4). A possible reason is how similar the digits look;
5. **The score definition can change the winner.** With the minimum of old and new instead of
   the mean, run C would pick 8 epochs · 0.00005 · 500,000 (79.1 % / 84.0 %) and run A
   8 epochs · 0.0001 · 1,000,000 (88.1 % / 87.7 %).

