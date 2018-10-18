"""Import all modules."""
from .ocdl_sgd import OnlineDictLearnSGD, OnlineDictLearnSGDMask
from .ocdl_surrogate_dense import OnlineDictLearnDenseSurrogate
from .ocdl_surrogate_slice import OnlineDictLearnSliceSurrogate

__all__ = [
    'OnlineDictLearnSGD',
    'OnlineDictLearnSGDMask',
    'OnlineDictLearnDenseSurrogate',
    'OnlineDictLearnSliceSurrogate',
]
