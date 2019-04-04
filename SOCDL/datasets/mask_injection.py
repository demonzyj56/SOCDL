"""Inject mask to dataset loader.
In general, we want deterministic masks for each image, but the L2Denoise can be
postponed until the images are requested.
"""
import logging
import numpy as np
import sporco.util as su
from .loader import DatasetLoader, BlobLoader

logger = logging.getLogger(__name__)


class MaskInjection(object):
    """Perform mask computation for each batch."""

    def __init__(self, loader, noise_fraction, transform=None):
        self.loader = loader
        self.noise_fraction = noise_fraction
        if transform is None:
            self.transform = lambda x, y: (x, y)
        else:
            self.transform = transform
        if isinstance(loader, BlobLoader):
            self.masks = su.rndmask(self.loader.blob.shape, self.noise_fraction,
                                    self.loader.blob.dtype)
        elif isinstance(loader, DatasetLoader):
            self.masks = [None for _ in range(len(self.loader))]
        else:
            raise TypeError('Unknown loader type.')
        self.rand_sample = self.loader.random_samples()

    def mask_from_index(self, index):
        """Generate mask."""
        if isinstance(self.loader, BlobLoader):
            masks = self.masks[..., index]
        elif isinstance(self.loader, DatasetLoader):
            for i in index:
                if self.masks[i] is None:
                    self.masks[i] = su.rndmask(self.rand_sample.shape,
                                               self.noise_fraction,
                                               self.rand_sample.dtype)
            masks = np.stack([self.masks[i] for i in index], axis=-1)
        else:
            raise TypeError('Unknown loader type.')
        return masks

    def random_samples(self):
        """Return random samples from loader."""
        return self.rand_sample

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, index):
        """Returns objects and masks."""
        # Transformation of loader is ignored
        return self.transform(self.loader.sample_from_index(index),
                              self.mask_from_index(index))

    def __iter__(self):
        self.loader = iter(self.loader)
        return self

    def __next__(self):
        """Reimplement loader.__next__."""
        try:
            index = next(self.loader.sampler)
        except Exception as e:
            raise e
        return self.transform(self.loader.sample_from_index(index),
                              self.mask_from_index(index))
