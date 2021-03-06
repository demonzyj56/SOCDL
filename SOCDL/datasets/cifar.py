"""Loader for CIFAR10/CIFAR100."""
import numpy as np
import torchvision.datasets.cifar as tcifar
from .loader import BlobLoader


class CIFAR10Loader(BlobLoader):
    """Load CIFAR10 images.  The labels are not loaded."""
    _cifar_func = tcifar.CIFAR10

    def __init__(self, root, epochs=None, batch_size=None, train=True,
                 dtype=np.float32, scaled=True, gray=False, transform=None):
        cifar_data = self._cifar_func(root, train=train, download=True)
        if train:
            blob = cifar_data.train_data
        else:
            blob = cifar_data.test_data
        blob = np.moveaxis(blob, 0, -1).astype(dtype, copy=True)
        if scaled and not np.issubdtype(blob.dtype, np.integer):
            blob /= 255.
        if gray:
            blob = blob[:, :, 0, :] * 0.2989 + blob[:, :, 1, :] * 0.5870 + \
                blob[:, :, 2, :] * 0.1140

        super(CIFAR10Loader, self).__init__(blob, epochs, batch_size, transform)


class CIFAR100Loader(CIFAR10Loader):
    """Load CIFAR100 images.  The labels are not loaded."""
    _cifar_func = tcifar.CIFAR100
