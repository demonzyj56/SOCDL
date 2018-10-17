"""Utilities for loading an image from path, with given options."""
import logging
import os
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def imread(filename, dtype=np.float32, scaled=True, gray=False, dsize=None):
    """Load image from filename.

    Parameters
    ----------
    filename: string
        File path for the image.
    dtype:
        Data type for the loaded image.
    scaled: boolean
        If scaled is True, then the image is scaled to the range [0., 1.],
        otherwise it remains in [0, 255].  Note that this option is valid only
        for floating data type.
    gray: boolean
        If True, then the image is converted to single channel grayscale image.
    dsize: tuple or None
        If not None, then the image is rescaled to this shape.

    Returns
    -------
    img: ndarray [H, W] (for grayscale image) or [H, W, C] (for color image)
        Image array with color channel at the last dimension.  If it is a color
        image, then the color channel is organized in RGB order.  If it is a
        grayscale image, then img is two dimensional.
    """
    assert os.path.exists(filename), \
        'Path does not exist: {}'.format(filename)
    if gray:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if dsize is not None:
        image = cv2.resize(image, dsize=dsize)
    image = image.astype(dtype)
    if scaled:
        if np.issubdtype(image.dtype, np.integer):
            logger.warning('Image type is %s, so is not scaled', image.dtype)
        else:
            image /= 255.
    return image
