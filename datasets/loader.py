"""Abstraction of loader for different cases."""
import numpy as np


class EpochSampler(object):
    """An iterator that yields exact number of epochs, each with specified
    batch_size.  If the samples are exhausted, then loop to the front, with
    indices shuffled."""

    def __init__(self, size, epochs=None, batch_size=None):
        self.size = size
        self.epochs = epochs if epochs is not None else size
        self.batch_size = min(batch_size, self.size) \
            if batch_size is not None else 1
        self.e = 0
        self.cur = 0
        self.idx = np.random.permutation(self.size)

    def __len__(self):
        return self.size

    def __iter__(self):
        return self

    def __next__(self):
        if self.e < self.epochs:
            self.e += 1
            return self._get_next_minibatch()
        else:
            raise StopIteration

    def _get_next_minibatch(self):
        if self.cur + self.batch_size > self.size:
            self.idx = np.random.permutation(self.size)
            self.cur = 0
        selected = self.idx[self.cur:self.cur+self.batch_size]
        self.cur += self.batch_size
        return selected


class EpochLoader(object):
    """An iterator that yields exact number of epochs, each with specified
    batch_size.  If the samples are exhausted, then loop to the front, with
    indices shuffled."""

    def __init__(self, epochs=None, batch_size=None):
        self.epochs = epochs
        self.batch_size = batch_size

    def __len__(self):
        """To be implemented."""
        raise NotImplementedError

    def sample_from_index(self, index):
        """To be implemented."""
        raise NotImplementedError

    def __getitem__(self, index):
        return self.sample_from_index(np.array(index)).squeeze()

    def __iter__(self):
        self.sampler = EpochSampler(len(self), self.epochs, self.batch_size)
        return self

    def __next__(self):
        try:
            index = next(self.sampler)
        except Exception as e:
            raise e
        return self.sample_from_index(index)


class DatasetLoader(EpochLoader):
    """Read data from dataset."""

    def __init__(self, dataset, epochs=None, batch_size=None):
        super(DatasetLoader, self).__init__(epochs, batch_size)
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def sample_from_index(self, index):
        """Assume each sample in the dataset is an image with HWC layout."""
        sampled = [self.dataset[i] for i in index]
        return np.stack(sampled, axis=-1)


class BlobLoader(EpochLoader):
    """Read data from an already constructed blob.
    Assume the blob has HWN (for grayscale images) or HWCN (for color images)
    layout.
    """

    def __init__(self, blob, epochs=None, batch_size=None):
        super(BlobLoader, self).__init__(epochs, batch_size)
        self.blob = blob

    def __len__(self):
        return self.blob.shape[-1]

    def sample_from_index(self, index):
        return self.blob[..., index]