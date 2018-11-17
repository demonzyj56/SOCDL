"""Import all modules."""
from sporco.dictlrn.onlinecdl import OnlineConvBPDNDictLearn
from .ocdl_sgd import OnlineDictLearnSGD, OnlineDictLearnSGDMask
from .ocdl_surrogate_dense import OnlineDictLearnDenseSurrogate
from .ocdl_surrogate_slice import OnlineDictLearnSliceSurrogate
from .ocdl_surrogate_slice_sampled import OnlineDictLearnSampledSliceSurrogate

__all__ = [
    'OnlineConvBPDNDictLearn',
    'OnlineDictLearnSGD',
    'OnlineDictLearnSGDMask',
    'OnlineDictLearnDenseSurrogate',
    'OnlineDictLearnSliceSurrogate',
    'OnlineDictLearnSampledSliceSurrogate',
]
