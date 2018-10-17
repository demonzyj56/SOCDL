"""Loader for VOC2007 images."""
import os
import numpy as np
import torch.utils.data as data
from .utils import imread
from .loader import DatasetLoader


class VOC07Images(data.Dataset):
    """Pascal VOC 2007 images"""
    def __init__(self, root, train=True, dtype=np.float32, scaled=True,
                 gray=False, dsize=None):
        if train:
            img_list = os.path.join(root, 'VOC2007', 'ImageSets', 'Main', 'trainval.txt')
        else:
            img_list = os.path.join(root, 'VOC2007', 'ImageSets', 'Main', 'test.txt')
        assert os.path.exists(img_list), 'Path does not exists: {}'.format(img_list)
        with open(img_list, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
        img_path_template = os.path.join(root, 'VOC2007', 'JPEGImages', '%s.jpg')
        self.img_path = [img_path_template % l for l in lines]
        assert all([os.path.exists(p) for p in self.img_path])
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


class VOC07Loader(DatasetLoader):
    """Pascal VOC07 image loader."""

    def __init__(self, root, epochs=None, batch_size=None, train=True,
                 dtype=np.float32, scaled=True, gray=False, dsize=(256, 256)):
        dataset = VOC07Images(root, train, dtype, scaled, gray, dsize)
        super(VOC07Loader, self).__init__(dataset, epochs, batch_size)
