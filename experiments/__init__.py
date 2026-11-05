# Makes `experiments` a Python package and re-exports its public classes, so that other
# files can simply write:   from experiments import MNISTPretrained, MNISTPretrainedConfig
#
# IMPORT ORDER MATTERS here. mnist_pretrained.py itself does
#     from experiments import Experiment, Config
# while this package is still being initialised. That only works because `.base` is
# imported FIRST (line below), so Experiment and Config already exist in this package
# when mnist_pretrained.py asks for them. If you swapped the two import lines you would
# get a circular-import error. Keep new experiments AFTER the `.base` import.
# mnist_ewc_usecase1.py and mnist_class_incremental.py use MNISTPretrained, so they come after it.
from .base import Experiment, Config
from .mnist_pretrained import MNISTPretrained, MNISTPretrainedConfig
from .mnist_ewc_usecase1 import MNISTEWC, MNISTEWCConfig
from .mnist_class_incremental import MNISTClassIncremental, MNISTClassIncrementalConfig


# Public API of the package (what `from experiments import *` would export).
__all__ = [
    "Experiment",
    "Config",
    "MNISTPretrained",
    "MNISTPretrainedConfig",
    "MNISTEWC",
    "MNISTEWCConfig",
    "MNISTClassIncremental",
    "MNISTClassIncrementalConfig",
]
