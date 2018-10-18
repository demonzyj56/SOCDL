"""Utility functions."""
import logging
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2

logger = logging.getLogger(__name__)


def setup_logging(filename=None):
    """Utility for every script to call on top-level.
    If filename is not None, then also log to the filename."""
    FORMAT = '[%(levelname)s %(asctime)s] %(filename)s:%(lineno)4d: %(message)s'
    DATEFMT = '%Y-%m-%d %H:%M:%S'
    logging.root.handlers = []
    handlers = [logging.StreamHandler(stream=sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename, mode='w'))
    logging.basicConfig(
        level=logging.INFO,
        format=FORMAT,
        datefmt=DATEFMT,
        handlers=handlers
    )


def Pcn(D, zm=True, gt=True):
    """Constraint set projection function that normalizes each dictionary
    to unary norm.

    Parameters
    ----------
    D: array
        Input dictionary to normalize.
    zm: bool
        If true, then the columns are mean subtracted before normalization.
    gt: bool
        If true, then only the entries of norms that are greater than 1
        is normalized.
    """
    # sum or average over all but last axis
    axis = tuple(range(D.ndim-1))
    if zm:
        D -= np.mean(D, axis=axis, keepdims=True)
    norm = np.sqrt(np.sum(D**2, axis=axis, keepdims=True))
    norm[norm == 0] = 1.
    if gt:
        norm[norm < 1.] = 1.
    return np.asarray(D / norm, dtype=D.dtype)


def einsum(subscripts, operands):
    """Wrapper around possible implementations of einsum."""
    operands = [torch.from_numpy(o) for o in operands]
    out = torch.einsum(subscripts, operands).numpy()
    return out


def im2slices(S, kernel_h, kernel_w, boundary='circulant_back'):
    r"""Convert the input signal :math:`S` to a slice form.
    Assuming the input signal having a standard shape as pytorch variable
    (N, C, H, W).  The output slices have shape
    (batch_size, slice_dim, num_slices_per_batch).
    """
    pad_h, pad_w = kernel_h - 1, kernel_w - 1
    S_torch = globals()['_pad_{}'.format(boundary)](S, pad_h, pad_w)
    with torch.no_grad():
        S_torch = torch.from_numpy(S_torch)  # pylint: disable=no-member
        slices = F.unfold(S_torch, kernel_size=(kernel_h, kernel_w))
    return slices.numpy()


def slices2im(slices, kernel_h, kernel_w, output_h, output_w,
              boundary='circulant_back'):
    r"""Reconstruct input signal :math:`\hat{S}` for slices.
    The input slices should have compatible size of
    (batch_size, slice_dim, num_slices_per_batch), and the
    returned signal has shape (N, C, H, W) as standard pytorch variable.
    """
    pad_h, pad_w = kernel_h - 1, kernel_w - 1
    with torch.no_grad():
        slices_torch = torch.from_numpy(slices)  # pylint: disable=no-member
        S_recon = F.fold(
            slices_torch, (output_h+pad_h, output_w+pad_w), (kernel_h, kernel_w)
        )
    S_recon = globals()['_crop_{}'.format(boundary)](
        S_recon.numpy(), pad_h, pad_w
    )
    return S_recon


def _pad_circulant_front(blob, pad_h, pad_w):
    """Pad a 4-D blob with circulant boundary condition at the front."""
    return np.pad(blob, ((0, 0), (0, 0), (pad_h, 0), (pad_w, 0)), 'wrap')


def _pad_circulant_back(blob, pad_h, pad_w):
    """Pad a 4-D blob with circulant boundary condition at the back."""
    return np.pad(blob, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), 'wrap')


def _pad_zeros_front(blob, pad_h, pad_w):
    """Pad a 4-D blob with zero boundary condition at the front."""
    return np.pad(blob, ((0, 0), (0, 0), (pad_h, 0), (pad_w, 0)), 'constant')


def _pad_zeros_back(blob, pad_h, pad_w):
    """Pad a 4-D blob with zero boundary condition at the back."""
    return np.pad(blob, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), 'constant')


def _crop_circulant_front(blob, pad_h, pad_w):
    """Crop a 4-D blob which is reconstructed for circulant boundary condition
    at the front."""
    cropped = blob[:, :, pad_h:, pad_w:]
    cropped[:, :, -pad_h:, :] += blob[:, :, :pad_h, pad_w:]
    cropped[:, :, :, -pad_w:] += blob[:, :, pad_h:, :pad_w]
    cropped[:, :, -pad_h:, -pad_w:] += blob[:, :, :pad_h, :pad_w]
    return cropped


def _crop_circulant_back(blob, pad_h, pad_w):
    """Crop a 4-D blob which is reconstructed for circulant boundary condition
    at the back."""
    cropped = blob[:, :, :-pad_h, :-pad_w]
    cropped[:, :, :pad_h, :] += blob[:, :, -pad_h:, :-pad_w]
    cropped[:, :, :, :pad_w] += blob[:, :, :-pad_h, -pad_w:]
    cropped[:, :, :pad_h, :pad_w] += blob[:, :, -pad_h:, -pad_w:]
    return cropped


def _crop_zeros_front(blob, pad_h, pad_w):
    """Crop a 4-D blob which is reconstructed for zero boundary condition
    at the front."""
    return blob[:, :, pad_h:, pad_w:]


def _crop_zeros_back(blob, pad_h, pad_w):
    """Crop a 4-D blob which is reconstructed for zero boundary condition
    at the back."""
    return blob[:, :, :-pad_h, :-pad_w]


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
