"""Transformation for images."""
import logging
import numpy as np
import sporco.util as su
from sporco.admm import tvl2

logger = logging.getLogger(__name__)


def default_transform(blob, pad_size=None, tikhonov=True):
    if pad_size is not None:
        pad = [(pad_size, pad_size), (pad_size, pad_size)] + \
            [(0, 0) for _ in range(blob.ndim-2)]
        blob = np.pad(blob, pad, mode='constant')
    if tikhonov:
        # fix lambda to be 5
        sl, sh = su.tikhonov_filter(blob, 5.)
    else:
        sl, sh = 0, blob
    return sl, sh


def masked_transform(blob, pad_size=None, noise_fraction=0.5, l2denoise=True,
                     gray=False):
    mask = su.rndmask(blob.shape, noise_fraction, dtype=blob.dtype)
    blobw = blob * mask
    if pad_size is not None:
        pad = [(pad_size, pad_size), (pad_size, pad_size)] + \
            [(0, 0) for _ in range(blob.ndim-2)]
        blobw = np.pad(blobw, pad, mode='constant')
        mask = np.pad(mask, pad, 'constant')
    if l2denoise:
        tvl2opt = tvl2.TVL2Denoise.Options({
            'Verbose': False, 'MaxMainIter': 200, 'gEvalY': False,
            'AutoRho': {'Enabled': True}, 'DFidWeight': mask
        })
        denoiser = tvl2.TVL2Denoise(blobw, 0.05, tvl2opt,
                                    caxis=None if gray else 2)
        sl = denoiser.solve()
        sh = mask * (blobw - sl)
    else:
        sl, sh = 0, blobw
    return sl, sh, mask
