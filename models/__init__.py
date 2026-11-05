# Makes `models` a Python package and re-exports the CNN class, so other files can write
#     from models import CNN
# instead of the longer
#     from models.cnn import CNN
from .cnn import CNN

# __all__ defines what `from models import *` exports (and documents the public API).
__all__ = ["CNN"]