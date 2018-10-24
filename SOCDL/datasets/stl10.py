"""Loader for STL10 dataset."""
import logging
import os
import numpy as np
from .loader import BlobLoader

logger = logging.getLogger(__name__)


class STL10Loader(BlobLoader):
    """Load STL10 images.  The labels are not loaded."""

    def __init__(self, root, epochs=None, batch_size=None, dtype=np.float32,
                 scaled=True, gray=False, transform=None, category='train'):
        assert category in ('train', 'test', 'unlabeled'), \
            'Unknown STL10 data split: %s' % category
        path = os.path.join(root, '{}_X.bin'.format(category))
        with open(path, 'rb') as f:
            blob = np.fromfile(f, dtype=np.uint8)
        blob = blob.reshape(-1, 3, 96, 96).transpose((2, 3, 1, 0)).astype(dtype)
        if scaled and not np.issubdtype(blob.dtype, np.integer):
            blob /= 255.
        if gray:
            blob = blob[:, :, 0, :] * 0.2989 + blob[:, :, 1, :] * 0.5870 + \
                   blob[:, :, 2, :] * 0.1140

        super(STL10Loader, self).__init__(blob, epochs, batch_size, transform)