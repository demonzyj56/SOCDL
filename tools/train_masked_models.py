#!/usr/bin/env python
"""Train online models with masks.
Run the following algorithms:
    - K-SVD
    - Batch-CDL
    - OCDL-SGD
    - SOCDL
"""
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
import torch
import sporco.metric as sm
from sporco.admm import cbpdn
import cv2

# add SOCDL working directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from SOCDL.configs.configs import cfg, merge_cfg_from_file, merge_cfg_from_list
from SOCDL.builder import get_online_solvers, get_loader, \
    snapshot_solver_dict, snapshot_solver_stats
from SOCDL.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Training a noise-free model.')
    parser.add_argument('--cfg', dest='cfg_file', default=None, type=str,
                        help='cfg file to parse options from')
    parser.add_argument('--def', dest='def_file', default=None, type=str,
                        help='cfg file that defines configs for solvers')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Command line arguments')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def train_models(defs):
    """Model training routine.  Several difference with train_models:
        - `defs` contains both train and test.
        - We train batch model as well.
        - The solvers are trained and tested on the same batch of images."""
    # initialize D0
    if cfg.TRAIN.DATASET.GRAY:
        D0 = np.random.randn(cfg.PATCH_SIZE, cfg.PATCH_SIZE,
                             cfg.NUM_ATOMS).astype(np.float32)
    else:
        D0 = np.random.randn(cfg.PATCH_SIZE, cfg.PATCH_SIZE, 3,
                             cfg.NUM_ATOMS).astype(np.float32)
    if not cfg.TRAIN.DATASET.TIKHONOV:
        D0[..., 0] = 1. / D0[..., 0].size

    # initialize loader
    loader = get_loader(train=True)
    init_sample = loader.random_samples()

    # initialize solvers
    solvers = get_online_solvers(defs['TRAIN'], D0, init_sample)

    for e, (slw, shw, blob, mask) in enumerate(loader):
        for k, s in solvers.items():
            s.solve(shw, mask)
            # For online solvers we explicitly save the dicts here.
            if cfg.SNAPSHOT:
                path = os.path.join(cfg.OUTPUT_PATH, k)
                snapshot_solver_dict(s, path, cur_cnt=e)
            # compute psnr and reconstruction
            opt = cbpdn.ConvBPDN.Options(defs['TEST']['ConvBPDN'])
            D = s.getdict().squeeze()
            ams = cbpdn.AddMaskSim(cbpdn.ConvBPDN, D, shw, mask, cfg.LAMBDA, opt=opt)
            ams.solve()
            shr = ams.reconstruct().squeeze()
            imgr = slw + shr.reshape(slw.shape)
            if cfg.TRAIN.DATASET.PAD_BOUNDARY:
                ps = cfg.PATCH_SIZE // 2
                imgr = imgr[ps:-ps, ps:-ps, ...]
            assert blob.shape == imgr.shape
            psnr = sm.psnr(blob, imgr, rng=1.)
            if cfg.VERBOSE:
                logger.info('PSNR for {} at iteration {} is {:.2f}'.format(
                    k, e, psnr
                ))
            if cfg.SNAPSHOT and cfg.TRAIN.BATCH_SIZE == 1:
                imgr = imgr.squeeze()
                imgr[imgr < 0] = 0
                imgr[imgr > 1] = 1
                imgr = (imgr*255).astype(np.uint8)
                imgr = cv2.cvtColor(imgr, cv2.COLOR_RGB2BGR)
                filename = '{}_{:.2f}.png'.format(e, psnr)
                filename= os.path.join(path, filename)
                cv2.imwrite(filename, imgr)

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
