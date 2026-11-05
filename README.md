# Continual Learning on MNIST with EWC

This project studies **Elastic Weight Consolidation (EWC)** on MNIST with a small CNN in two
use cases. Experience Replay is handled in a separate part of the project; this part uses
**EWC only, without replay**.

## What EWC does

After training on old data, EWC stores the weights θ* and estimates how important each weight
is (Fisher information F). While training on new data, it adds a penalty to the loss:

```
L(θ) = L_new(θ) + (λ / 2) · Σᵢ Fᵢ · (θᵢ − θ*ᵢ)²
```

| Symbol | Meaning |
|---|---|
| L_new | cross-entropy loss on the new data |
| θᵢ, θ*ᵢ | current value and value after the old training of weight i |
| Fᵢ | importance of weight i for the old data (mean squared gradient of the log-likelihood) |
| λ | EWC strength; λ = 0 means plain fine-tuning (the baseline) |

Large λ protects old knowledge (stability) but makes it harder to learn new data (plasticity).

## Project structure

```
main.py                              Use case 2, step 1: pretrain on few images per class
main_ewc.py                          Use case 2, step 2: continue training with EWC (several λ)
main_usecase1.py                     Use case 1: add a new class with EWC (framework version)
usecase1.py                          Use case 1 as a single file (same procedure, no framework)
experiments/base.py                  Experiment skeleton (seed, data, train, save, evaluate)
experiments/mnist_pretrained.py      Pretraining experiment (data selection, training loop)
experiments/mnist_ewc.py             Use case 2: EWC experiment
experiments/mnist_class_incremental.py  Use case 1: class-incremental EWC experiment
models/cnn.py                        CNN: 2 conv blocks + 2 linear layers (206,922 parameters)
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.13:

```
uv sync
```

MNIST is downloaded automatically into `.data`. Trained weights are written to `.out`.

## Use case 1: add a new class

The model is trained on a few classes (default: 0 and 3) and then trained further on a new
class (default: 8) **only**, so the old classes are at risk of being forgotten.

* Step 1 data: old classes from one random half of the MNIST training set
* Step 2 data: new classes from the other half (no old-class images, no replay)
* Test data: all MNIST test images of the old and new classes

Step 1 is trained once. Then, starting from the same step-1 model, the new class is learned
twice: with plain fine-tuning (baseline) and with EWC. The accuracy on the old and on the new
classes is printed for both.

Run:

```
python main_usecase1.py
```

All settings (classes, images per class, λ, epochs, learning rate, seed) are at the top of
`main_usecase1.py`. `usecase1.py` does the same in a single file without the framework.
To check how robust a result is, run it again with different seeds.

## Use case 2: improve a  pretrained model

The model is pretrained on only 100 images per digit (1,000 images) and then trained further
on 1,000 **other** images per digit (10,000 images) with EWC.

* Test data: 200 test images per digit (2,000 images), the same in both steps

Run in this order (`main_ewc.py` loads the model saved by `main.py`):

```
python main.py
python main_ewc.py
```

`seed`, `samples_per_class` and `test_samples_per_class` must be identical in both scripts.
The "Pretrained model" accuracy printed by `main_ewc.py` must match the final accuracy of
`main.py`; this confirms that both use the same data.

## Results

Fill in after running the scripts.

**Use case 1** (old: 0, 3; new: 8; λ = 1,000,000)

| seed | method | old classes | new class |
|---|---|---|---|
| 42 | plain fine-tuning | | |
| 42 | EWC | | |

**Use case 2** (test accuracy after step 2; pretrained model: … %)

| λ | test accuracy |
|---|---|
| 0 (no EWC) | |
| 0.1 | |
| 1 | |
| 10 | |
| 100 | |

## Observations

* **Use case 1:** without EWC the old classes are forgotten completely (0 %). EWC preserves
  them to a large extent, at the cost of lower accuracy on the new class. EWC needs a large λ
  (around 10⁶) together with a small learning rate (0.0001), because the Fisher values of a
  very confident model are small. Results vary strongly between seeds.
* **Use case 2:** the new data comes from the same distribution as the pretraining data, so
  nothing is forgotten. EWC therefore cannot improve on plain fine-tuning: the accuracy is
  highest for λ = 0 and decreases as λ grows, because EWC ties the model to the weak
  pretrained weights.

## Limitations

* λ and the other settings were chosen by looking at the test accuracy; there is no
  separate validation set, so the reported numbers may be slightly optimistic.
* Results depend on the random seed (data selection, initial weights, batch order).
