"""Test runner."""
import functools
import logging
import multiprocessing as mp
import warnings
import numpy as np
from scipy import linalg
import sporco.linalg as spl
import sporco.metric as sm
from sporco.admm import cbpdn
try:
    import sporco_cuda.cbpdn as cucbpdn
except ImportError:
    cucbpdn = None
# Required due to pyFFTW bug #135 - see "Notes" section of SPORCO docs.
spl.pyfftw_threads = 1

from .configs import cfg as _cfg

logger = logging.getLogger(__name__)


def run_cbpdn(D, sl, sh, lmbda, test_blob=None, opt=None):
    """Run ConvBPDN solver and compute statistics."""
    opt = cbpdn.ConvBPDN.Options(opt)
    if test_blob is None:
        test_blob = sl + sh
    solver = cbpdn.ConvBPDN(D, sh, lmbda, opt=opt)
    solver.solve()
    fnc = solver.getitstat().ObjFun[-1]
    shr = solver.reconstruct().squeeze()
    imgr = sl + shr
    psnr = 0.
    for idx in range(sh.shape[-1]):
        psnr += sm.psnr(test_blob[..., idx], imgr[..., idx], rng=1.)
    psnr /= test_blob.shape[-1]
    return fnc, psnr


def run_cbpdn_gpu(D, sl, sh, lmbda, test_blob=None, opt=None):
    """Run GPU version of CBPDN.  Only supports grayscale images."""
    assert _cfg.DATASET.GRAY, 'Only grayscale images are supported'
    assert cucbpdn is not None, 'GPU CBPDN is not supported'
    opt = cbpdn.ConvBPDN.Options(opt)
    if test_blob is None:
        test_blob = sl + sh
    fnc, psnr = 0., 0.
    for idx in range(test_blob.shape[-1]):
        X = cucbpdn.cbpdn(D, sh[..., idx].squeeze(), lmbda, opt=opt)
        shr = np.sum(spl.fftconv(D, X), axis=2)
        dfd = linalg.norm(shr.ravel()-sh[..., idx].ravel())
        rl1 = linalg.norm(X.ravel(), 1)
        obj = dfd + lmbda * rl1
        fnc += obj
        imgr = sl[..., idx] + shr
        psnr += sm.psnr(imgr, test_blob[..., idx].squeeze(), rng=1.)
    psnr /= test_blob.shape[-1]
    return fnc, psnr


def map_cbpdn_dicts(dicts, sl, sh, lmbda, test_blob=None, opt=None):
    """Run CBPDN in parallel over a series of dictionaries."""
    if test_blob is None:
        test_blob = sl + sh
    # catch deprecated warnings in pyfftw and ignore it
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if _cfg.GPU_TEST:
            results = [run_cbpdn_gpu(D, sl, sh, lmbda, test_blob, opt)
                       for D in dicts]
        else:
            with mp.Pool(_cfg.NUM_PROCESSES) as pool:
                results = pool.map(
                    functools.partial(run_cbpdn, sl=sl, sh=sh, lmbda=lmbda,
                                      test_blob=test_blob, opt=opt),
                    dicts
                )
    return results