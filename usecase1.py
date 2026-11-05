"""
Use case 1 (single-file version): train on a few classes, then ADD a new class with EWC.
One file, no framework needed except models/cnn.py. Useful for quick single runs.
The framework version (several seeds, mean and std) is main_usecase1.py.

Run from the project root:   python usecase1.py
"""
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import CNN

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

assert not set(OLD_CLASSES) & set(NEW_CLASSES), "old and new classes must be different"
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # GPU if available

# ---------------- data ----------------
train_ds = datasets.MNIST("./.data", train=True, download=True, transform=transforms.ToTensor())
test_ds = datasets.MNIST("./.data", train=False, download=True, transform=transforms.ToTensor())


def only_classes(ds, indices, classes, n_per_class=None):
    """Subset of ds: from the given indices keep only images whose label is in `classes`,
    at most `n_per_class` images per class (None = no limit).
    The indices are in random order, so the first n images of a class are a random choice."""
    labels = ds.targets.tolist()
    taken = {c: 0 for c in classes}        # how many images of each class were picked so far
    chosen = []
    for i in indices:
        c = labels[i]
        if c in taken and (n_per_class is None or taken[c] < n_per_class):
            chosen.append(i)
            taken[c] += 1
    return Subset(ds, chosen)


# Split the 60,000 training images randomly into two halves.
perm = torch.randperm(len(train_ds)).tolist()
half_a, half_b = perm[:30000], perm[30000:]     # half_a for step 1, half_b for step 2

data_a = only_classes(train_ds, half_a, OLD_CLASSES, SAMPLES_PER_OLD_CLASS)   # step 1: old classes
data_b = only_classes(train_ds, half_b, NEW_CLASSES, SAMPLES_PER_NEW_CLASS)   # step 2: new classes
# (to also use old-class images in step 2, write NEW_CLASSES + OLD_CLASSES above)


# ---------------- helpers ----------------
def train(model, data, epochs, ewc=None, lam=0.0):
    """Train `model` on `data`. If `ewc` is given, the EWC penalty (strength `lam`) is added."""
    loader = DataLoader(data, batch_size=BATCH, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y)
            if ewc is not None:
                loss = loss + (lam / 2) * ewc_penalty(model, ewc)   # EWC: stay close to old weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.no_grad()
def accuracy(model, classes):
    """Test accuracy (in %) on the test images of the given classes."""
    model.eval()
    loader = DataLoader(only_classes(test_ds, range(len(test_ds)), classes), batch_size=512)
    correct, total = 0, 0
    for x, y in loader:
        correct += (model(x.to(device)).argmax(dim=1).cpu() == y).sum().item()
        total += len(y)
    return 100 * correct / total


def compute_ewc(model, data, n_samples=2000):
    """Remember the old weights and how important each weight is (Fisher information).
    n_samples = how many images are used to estimate the importance."""
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    model.eval()
    count = 0
    for x, _ in DataLoader(data, batch_size=1, shuffle=True):
        model.zero_grad()
        log_probs = F.log_softmax(model(x.to(device)), dim=1)
        label = torch.multinomial(log_probs.exp(), 1).item()     # label sampled from the model
        (-log_probs[0, label]).backward()
        for name, p in model.named_parameters():
            fisher[name] += p.grad ** 2                          # squared gradient = importance
        count += 1
        if count >= n_samples:
            break
    model.zero_grad()
    return {
        "fisher": {name: f / count for name, f in fisher.items()},
        "old": {name: p.detach().clone() for name, p in model.named_parameters()},
    }


def ewc_penalty(model, ewc):
    """Sum of importance * (change of weight)^2 : big for important weights that moved."""
    return sum(
        (ewc["fisher"][name] * (p - ewc["old"][name]) ** 2).sum()
        for name, p in model.named_parameters()
    )


def report(title, model):
    print(
        f"{title:<28} old classes {OLD_CLASSES}: {accuracy(model, OLD_CLASSES):6.2f}% | "
        f"new classes {NEW_CLASSES}: {accuracy(model, NEW_CLASSES):6.2f}%"
    )


# ---------------- run ----------------
print(f"Step 1: train on classes {OLD_CLASSES} ({len(data_a)} images)")
base = CNN().to(device)
train(base, data_a, EPOCHS)
report("After step 1", base)

ewc = compute_ewc(base, data_a)

print(f"\nStep 2: add classes {NEW_CLASSES} ({len(data_b)} images)")

plain = copy.deepcopy(base)                  # same starting point for a fair comparison
train(plain, data_b, EPOCHS)
report("Plain fine-tuning (no EWC)", plain)

with_ewc = copy.deepcopy(base)
train(with_ewc, data_b, EPOCHS, ewc=ewc, lam=EWC_LAMBDA)
report(f"With EWC (lambda={EWC_LAMBDA})", with_ewc)