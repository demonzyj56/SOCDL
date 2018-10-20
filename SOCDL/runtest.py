"""Test runner."""
import functools
import logging
import multiprocessing as mp
import warnings
import sporco.linalg as sl
import sporco.metric as sm
from sporco.admm import cbpdn
# Required due to pyFFTW bug #135 - see "Notes" section of SPORCO docs.
sl.pyfftw_threads = 1

from .configs import cfg as _cfg

logger = logging.getLogger(__name__)


def run_cbpdn(D, sl, sh, lmbda, test_blob=None, opt=None):
    """Run ConvBPDN solver and compute statistics."""
    if opt is None:
        opt = cbpdn.ConvBPDN.Options()
    else:
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


def map_cbpdn_dicts(dicts, sl, sh, lmbda, test_blob=None, opt=None):
    """Run CBPDN in parallel over a series of dictionaries."""
    if test_blob is None:
        test_blob = sl + sh
    # catch deprecated warnings in pyfftw and ignore it
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with mp.Pool(_cfg.NUM_PROCESSES) as pool:
            results = pool.map(
                functools.partial(run_cbpdn, sl=sl, sh=sh, lmbda=lmbda,
                                  test_blob=test_blob, opt=opt),
                dicts
            )
    return results