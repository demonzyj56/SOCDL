#!/usr/bin/env python
"""Inpainting of selected single images.  Hard code everything."""
import argparse
import datetime
import logging
import os
import pprint
import sys
import warnings
import yaml
import pyfftw  # pylint: disable=unused-import
import numpy as np
import cv2
import torch
import sporco.util as su
import sporco.metric as sm
from sporco.admm import cbpdn, tvl2
from sporco.dictlrn import cbpdndlmd

# add SOCDL working directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from SOCDL.configs.configs import cfg, merge_cfg_from_file, merge_cfg_from_list
from SOCDL.builder import get_online_solvers, \
    snapshot_solver_dict, snapshot_solver_stats
from SOCDL.datasets.image_dataset import create_image_blob
from SOCDL.utils import setup_logging

logger = logging.getLogger(__name__)


def pad_func(blob):
    """Pad image when necessary."""
    if cfg.TRAIN.DATASET.PAD_BOUNDARY:
        pad = [(0, cfg.PATCH_SIZE-1), (0, cfg.PATCH_SIZE-1)] + \
            [(0, 0) for _ in range(blob.ndim-2)]
        blob = np.pad(blob, pad, mode='constant')
    return blob


def crop_func(blob):
    """Crop the padded image."""
    if cfg.TRAIN.DATASET.PAD_BOUNDARY:
        H = blob.shape[0] - cfg.PATCH_SIZE + 1
        W = blob.shape[1] - cfg.PATCH_SIZE + 1
        blob = blob[:H, :W, ...]
    return blob


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Train mask solvers on single images.')
    parser.add_argument('--cfg', dest='cfg_file', default=None, type=str,
                        help='cfg file to parse options from')
    parser.add_argument('--def', dest='def_file', default=None, type=str,
                        help='cfg file that defines configs for solvers')
    parser.add_argument('--img_name', default=None, type=str,
                        help='Define image name')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Command line arguments')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def reconstruct_ams(D, defs, slw, shw, img, mask, crop_func):
    """Reconstruct image using trained D and compute PSNR."""
    opt = cbpdn.ConvBPDN.Options(defs)
    ams = cbpdn.AddMaskSim(cbpdn.ConvBPDN, D, shw, mask, cfg.LAMBDA, opt=opt)
    ams.solve()
    shr = ams.reconstruct().squeeze()
    imgr = slw + shr.reshape(slw.shape)
    imgr = crop_func(imgr)
    assert img.shape == imgr.shape
    psnr = sm.psnr(img, imgr, rng=1.)
    return imgr, psnr


def decompose_image(img, mask):
    imgw = img * mask
    imgw = pad_func(imgw)
    maskw = pad_func(mask)
    if cfg.MASK.L2DENOISE:
        tvl2opt = tvl2.TVL2Denoise.Options({
            'Verbose': False, 'MaxMainIter': 200, 'gEvalY': False,
            'AutoRho': {'Enabled': True}, 'DFidWeight': maskw
        })
        denoiser = tvl2.TVL2Denoise(imgw, 0.05, tvl2opt,
                                    caxis=None if cfg.TRAIN.DATASET.GRAY else 2)
        sl = denoiser.solve()
        sh = imgw - sl
        if cfg.VERBOSE:
            logger.info('L2Denoise PSNR: {:.2f}'.format(
                sm.psnr(img, crop_func(sh*maskw+sl), rng=1.)
            ))
    else:
        sl, sh = np.zeros_like(imgw), imgw
    return sl, sh, maskw


def save_image(img, path):
    img = img.squeeze()
    img[img < 0] = 0
    img[img > 1] = 1
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def train_online_solver(D0, train_defs, test_defs, slw, shw, img, mask, crop_func):
    # Note that img is 4-dim.
    key = list(train_defs.keys())[0]  # only one key is available
    solver = get_online_solvers(train_defs, D0, shw)[key]
    path = os.path.join(cfg.OUTPUT_PATH, key)
    for e in range(cfg.TRAIN.EPOCHS):
        solver.solve(shw, mask)
        imgr, psnr = reconstruct_ams(solver.getdict().squeeze(), test_defs,
                                     slw, shw, img, mask, crop_func)
        img_path = os.path.join(path, '{:d}_{:.2f}.png'.format(e, psnr))
        if cfg.SNAPSHOT:
            snapshot_solver_dict(solver, path, cur_cnt=e)
            save_image(imgr, img_path)
        if cfg.VERBOSE:
            logger.info('PSNR for {} at iteration {} is {:.2f}'.format(
                key, e, psnr
            ))
    return {key: solver}

def trainConvBPDNMaskDictLearn(D0, train_defs, test_defs, slw, shw, img, mask, crop_func):
    key = list(train_defs.keys())[0]
    path = os.path.join(cfg.OUTPUT_PATH, key)
    os.makedirs(path, exist_ok=True)

    def _callback(d):
        """Snapshot dictionaries for every iteration."""
        imgr, psnr = reconstruct_ams(d.getdict().squeeze(), test_defs,
                                     slw, shw, img, mask, crop_func)
        img_path = os.path.join(path, '{:d}_{:.2f}.png'.format(d.j, psnr))
        if cfg.SNAPSHOT:
            snapshot_solver_dict(d, path)
            save_image(imgr, img_path)
        if cfg.VERBOSE:
            logger.info('PSNR for {} at iteration {} is {:.2f}'.format(
                key, d.j, psnr
            ))
        return 0

    opts = train_defs['ConvBPDNMaskDictLearn']
    opts.update({'Callback': _callback, 'MaxMainIter': cfg.TRAIN.EPOCHS})
    mdopt = cbpdndlmd.ConvBPDNMaskDictLearn.Options(opts, dmethod='cns')
    solver = cbpdndlmd.ConvBPDNMaskDictLearn(D0, shw, cfg.LAMBDA, mask,
                                             opt=mdopt, dmethod='cns')
    solver.solve()
    return {key: solver}


def trainKSVD(D0, train_defs, test_defs, slw, shw, img, mask, crop_func):
    raise NotImplementedError


def train_models(defs):
    # initialize D0
    if cfg.TRAIN.DATASET.GRAY:
        D0 = np.random.randn(cfg.PATCH_SIZE, cfg.PATCH_SIZE,
                             cfg.NUM_ATOMS).astype(np.float32)
    else:
        D0 = np.random.randn(cfg.PATCH_SIZE, cfg.PATCH_SIZE, 3,
                             cfg.NUM_ATOMS).astype(np.float32)
    if not cfg.TRAIN.DATASET.TIKHONOV:
        D0[..., 0] = 1. / D0[..., 0].size
    # get image and mask
    img = create_image_blob(cfg.TRAIN.DATASET.IMAGE_NAMES[0])
    mask = su.rndmask(img.shape, cfg.MASK.NOISE, img.dtype)
    if cfg.VERBOSE:
        logger.info('Corrupted image PSNR: {:.2f}'.format(
            sm.psnr(img, img*mask, rng=1.)
        ))
    if cfg.SNAPSHOT:
        save_image(img*mask, os.path.join(
            cfg.OUTPUT_PATH,
            '{}_corrupted.png'.format(cfg.TRAIN.DATASET.IMAGE_NAMES[0])
        ))
    slw, shw, maskw = decompose_image(img, mask)
    socdl_def = defs['TRAIN'].get('OnlineDictLearnSliceSurrogate', None)
    ocdl_sgd_def = defs['TRAIN'].get('OnlineDictLearnSGDMask', None)
    cbpdndlmd_def = defs['TRAIN'].get('ConvBPDNMaskDictLearn', None)
    ksvd_def = defs['TRAIN'].get('KSVD', None)
    solvers = {}
    if socdl_def is not None:
        socdl = train_online_solver(
            D0, {'OnlineDictLearnSliceSurrogate': socdl_def},
            defs['TEST']['ConvBPDN'], slw, shw, img, maskw, crop_func
        )
        solvers.update(socdl)
    if ocdl_sgd_def is not None:
        ocdl_sgd = train_online_solver(
            D0, {'OnlineDictLearnSGDMask': ocdl_sgd_def},
            defs['TEST']['ConvBPDN'], slw, shw, img, maskw, crop_func
        )
        solvers.update(ocdl_sgd)
    if cbpdndlmd_def is not None:
        cbpdndlmd_dict = trainConvBPDNMaskDictLearn(
            D0, {'ConvBPDNMaskDictLearn': cbpdndlmd_def},
            defs['TEST']['ConvBPDN'], slw, shw, img, maskw, crop_func
        )
        solvers.update(cbpdndlmd_dict)
    if ksvd_def is not None:
        ksvd = trainKSVD(
            D0, {'KSVD': ksvd_def},
            defs['TEST']['ConvBPDN'], slw, shw, img, maskw, crop_func
        )
        solvers.update(ksvd)

    return solvers


def main():
    """Main entry."""
    args = parse_args()
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    if not os.path.exists(cfg.OUTPUT_PATH):
        os.makedirs(cfg.OUTPUT_PATH)
    if args.img_name is not None:
        cfg.TRAIN.DATASET.IMAGE_NAMES = [args.img_name]
    log_name = os.path.join(
        cfg.OUTPUT_PATH,
        '{:s}.{:%Y-%m-%d_%H-%M-%S}.train.log'.format(
            cfg.NAME,
            datetime.datetime.now(),
        )
    )
    setup_logging(log_name)
    logger.info('*' * 49)
    logger.info('* DATE: %s', str(datetime.datetime.now()))
    logger.info('*' * 49)
    logger.info('Called with args:')
    logger.info(args)
    logger.info('Training with config:')
    logger.info(pprint.pformat(cfg))
    if cfg.RNG_SEED >= 0:
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
    with open(args.def_file, 'r') as f:
        defs = yaml.load(f)
    logger.info('Solver definition:')
    logger.info(pprint.pformat(defs))

    solvers = train_models(defs)
    if cfg.SNAPSHOT:
        for k, v in solvers.items():
            snapshot_solver_stats(v, os.path.join(cfg.OUTPUT_PATH, k))


if __name__ == "__main__":
    # surpress all warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
