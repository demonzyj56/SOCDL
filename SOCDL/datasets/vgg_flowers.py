"""Loader for VGG Flower 17/102 dataset."""
import os
import numpy as np
import scipy.io as sio
import torch.utils.data as data
from .utils import imread
from .loader import DatasetLoader


class VGG17Flowers(data.Dataset):
    """VGG 17Flowers dataset."""

    def __init__(self, root, train=True, dtype=np.float32, scaled=True,
                 gray=False, dsize=None):
        with open(os.path.join(root, 'files.txt'), 'r') as f:
            lines = [l.strip() for l in f.readlines()]
            allfiles = [os.path.join(root, l) for l in lines]
        assert all([os.path.exists(f) for f in allfiles])
        mat = sio.loadmat(os.path.join(root, 'datasplits.mat'))
        if train:
            fileidx = mat['trn1'].ravel().tolist() + mat['val1'].ravel().tolist()
        else:
            fileidx = mat['tst1'].ravel().tolist()
        self.img_path = [allfiles[idx-1] for idx in fileidx]
        self.train = train
        self.dtype = dtype
        self.scaled = scaled
        self.gray = gray
        self.dsize = dsize

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        return imread(self.img_path[index], dtype=self.dtype,
                      scaled=self.scaled, gray=self.gray, dsize=self.dsize)


class VGG102Flowers(data.Dataset):
    """VGG 102Flowers dataset."""

    def __init__(self, root, train=True, dtype=np.float32, scaled=True,
                 gray=False, dsize=None):
        allfiles = filter(lambda s: s.endswith('jpg'), os.listdir(root))
        allfiles = [os.path.join(root, f) for f in allfiles]
        assert all([os.path.exists(f) for f in allfiles])
        mat = sio.loadmat(os.path.join(root, 'setid.mat'))
        if train:
            fileidx = mat['trnid'].ravel().tolist() + mat['valid'].ravel().tolist()
        else:
            fileidx = mat['tstid'].ravel().tolist()
        self.img_path = [allfiles[idx-1] for idx in fileidx]
        self.train = train
        self.dtype = dtype
        self.scaled = scaled
        self.gray = gray
        self.dsize = dsize

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        return imread(self.img_path[index], dtype=self.dtype,
                      scaled=self.scaled, gray=self.gray, dsize=self.dsize)


class VGG17FlowersLoader(DatasetLoader):
    """VGG17 flowers image loader."""

    def __init__(self, root, epochs=None, batch_size=None, train=True,
                 dtype=np.float32, scaled=True, gray=False, dsize=(256, 256)):
        dataset = VGG17Flowers(root, train, dtype, scaled, gray, dsize)
        super(VGG17FlowersLoader, self).__init__(dataset, epochs, batch_size)


class VGG102FlowersLoader(DatasetLoader):
    """VGG102 flowers image loader."""

    def __init__(self, root, epochs=None, batch_size=None, train=True,
                 dtype=np.float32, scaled=True, gray=False, dsize=(256, 256)):
        dataset = VGG102Flowers(root, train, dtype, scaled, gray, dsize)
        super(VGG102FlowersLoader, self).__init__(dataset, epochs, batch_size)
