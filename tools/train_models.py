#!/usr/bin/env python
"""Train online models without masks."""
import argparse
import datetime
import logging
import os
import pprint
import sys
import yaml
import pyfftw  # pylint: disable=unused-import
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from SOCDL.configs.configs import cfg, merge_cfg_from_file, merge_cfg_from_list
from SOCDL.builder import get_online_solvers, get_loader
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
    """Model training routine."""
    # initialize D0
    if cfg.DATASET.GRAY:
        D0 = np.random.randn(cfg.PATCH_SIZE, cfg.PATCH_SIZE,
                             cfg.NUM_ATOMS).astype(np.float32)
    else:
        D0 = np.random.randn(cfg.PATCH_SIZE, cfg.PATCH_SIZE, 3,
                             cfg.NUM_ATOMS).astype(np.float32)
    if not cfg.DATASET.TIKHONOV:
        D0[..., 0] = 1. / D0[..., 0].size

    # initialize loader
    loader = get_loader(train=True)
    init_sample = loader.random_samples()[1]

    # initialize solvers
    solvers = get_online_solvers(defs, D0, init_sample)

    for e, (_, sh) in enumerate(loader):
        for k, s in solvers.items():
            s.solve(sh)
            # For online solvers we explicitly save the dicts here.
            if cfg.SNAPSHOT:
                path = os.path.join(cfg.OUTPUT_PATH, k,
                                    '{}.{}.npy'.format(cfg.DATASET.NAME, e))
                np.save(path, s.getdict().squeeze())

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
        '{:s}.{:%Y-%m-%d_%H-%M-%S}.log'.format(
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
    # setup seeds and data types from cfg
    if cfg.RNG_SEED >= 0:
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
    with open(args.def_file, 'r') as f:
        defs = yaml.load(f)
    logger.info('Solver definition:')
    logger.info(pprint.pformat(defs))

    solvers = train_models(defs)


if __name__ == "__main__":
    main()
