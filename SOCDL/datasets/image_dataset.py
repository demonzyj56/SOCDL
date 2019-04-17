"""Create dataset for images."""
import logging
import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from .loader import BlobLoader, DatasetLoader
from ..utils import imread

logger = logging.getLogger(__name__)


THIS_DIR = os.path.dirname(__file__)


IMAGE_LIST = {
    'lena': 'images/ece533/lena.ppm',
    'barbara': 'images/sporco_get_images/standard/barbara.bmp',
    'boat.gray': 'images/misc/boat.512.tiff',
    'house': 'images/misc/4.1.05.tiff',
    'peppers': 'images/misc/4.2.07.tiff',
    'cameraman.gray': 'images/ece533/cameraman.tif',
    'man.gray': 'images/sporco_get_images/standard/man.grey.tiff',
    'mandrill': 'images/sporco_get_images/standard/mandrill.tiff',
    'monarch': 'images/sporco_get_images/standard/monarch.png',
    'fruit': ['images/FCSC/Images/fruit_100_100/{}.jpg'.format(i+1)
              for i in range(10)],
    'city': ['images/FCSC/Images/city_100_100/{}.jpg'.format(i+1)
             for i in range(10)],
    'singles': [  # used as test images for fruit and city
        'images/FCSC/Images/singles/test1/test1.jpg',
        'images/FCSC/Images/singles/test2/test2.jpg',
        'images/FCSC/Images/singles/test3/test3.jpg',
        'images/FCSC/Images/singles/test4/test.jpg',
    ],
    'f16': 'images/classic/f16.tif',
    'cat': 'images/ece533/cat.png',
    'watch': 'images/ece533/watch.png',
    'boy': 'images/ece533/boy.bmp',
    'sails': 'images/ece533/sails.png',
    'girl': 'images/ece533/girl.png',
    'tulips': 'images/ece533/tulips.png',
    'parrots': 'images/sporco_get_images/kodak/kodim23.png',
    'castle': 'images/KSVD/castle.png',
    'duck': 'images/KSVD/duck256.png',
    'horses': 'images/KSVD/horses.png',
    'kangaroo': 'images/KSVD/kangaroo.png',
    'mushroom': 'images/KSVD/mushroom.png',
    'train': 'images/KSVD/train.png',
    'zurich': 'images/KSVD/zurich256.png',
    'beacon': 'images/sporco_get_images/kodak/kodim19.png',
}


def create_image_blob(name, dtype=np.float32, scaled=True, gray=False,
                      dsize=None):
    """Create a 3/4-D image blob from a valid image name.

    Parameters
    ----------
    name: string
        The name of image or image list.
    dtype, scaled, gray, dsize
        Refer to `image_dataset.load_image` for more details.

    Returns
    -------
    blob: ndarray, [H, W, N] or [H, W, C, N]
        Loaded 3/4-D image blob.  The dimension is organized as [height, width,
        channels, batch], which is for the ease of internal use of SPORCO.
    """
    img_list = IMAGE_LIST[name]
    if not isinstance(img_list, list):
        img_list = [img_list]
    img_list = [os.path.join(THIS_DIR, name) for name in img_list]
    imgs = [imread(img, dtype, scaled, gray, dsize) for img in img_list]
    return np.stack(imgs, axis=-1)


class ImageDataset(Dataset):
    """A torch Dataset object that holds a series of images."""

    def __init__(self, names, dtype=np.float32, scaled=True, gray=False,
                 dsize=None):
        if not isinstance(names, list):
            names = [names]
        img_list = [os.path.join(THIS_DIR, IMAGE_LIST[n]) for n in names]
        self.imgs = [imread(img, dtype=dtype, scaled=scaled, gray=gray,
                            dsize=dsize) for img in img_list]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        return self.imgs[index]


class FruitLoader(BlobLoader):
    """Load dataset `Fruit`."""
    _name = 'fruit'

    def __init__(self, epochs=None, batch_size=None, train=True,
                 dtype=np.float32, scaled=True, gray=True, use_processed=False,
                 transform=None):
        """Load fruit or city dataset.  If use_processed is True, then use
        the processed data from `OCSC`."""
        if use_processed:
            assert gray, 'Only grayscale images are provided by OCSC'
            tt = 'train' if train else 'test'
            path = os.path.join(THIS_DIR, 'images', 'OCSC',
                                '{}_10'.format(self._name), tt,
                                '{}_lcne.mat'.format(tt))
            blob = sio.loadmat(path)['b'].astype(dtype)
        else:
            name = self._name if train else 'singles'
            blob = create_image_blob(name, dtype=dtype, scaled=scaled,
                                     gray=gray)
        super(FruitLoader, self).__init__(blob, epochs, batch_size, transform)


class CityLoader(FruitLoader):
    """Load dataset `City`."""
    _name = 'city'


class ImageLoader(DatasetLoader):
    """Loader for a series of image names."""

    def __init__(self, names, epochs=None, batch_size=None,
                 dtype=np.float32, scaled=True, gray=False, dsize=None,
                 transform=None):
        dataset = ImageDataset(names, dtype, scaled, gray, dsize)
        super(ImageLoader, self).__init__(dataset, epochs, batch_size,
                                          transform)
