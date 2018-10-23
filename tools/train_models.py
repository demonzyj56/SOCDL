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
import sporco.util as su

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
    parser.add_argument('--run_test', action='store_true',
                        help='Also run test immediately after training')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Command line arguments')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def train_models(defs):
    """Model training routine."""
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
    init_sample = loader.random_samples()[1]

    # initialize solvers
    solvers = get_online_solvers(defs, D0, init_sample)

    for e, (_, sh) in enumerate(loader):
        for k, s in solvers.items():
            s.solve(sh)
            # For online solvers we explicitly save the dicts here.
            if cfg.SNAPSHOT:
                path = os.path.join(cfg.OUTPUT_PATH, k)
                snapshot_solver_dict(s, path, cur_cnt=e)

    return solvers


def visualize_dicts(solvers):
    """Show visualizations of learned dictionaries."""
    try:
        import matplotlib.pyplot as plt
    except:
        logger.warning('plt is not available, thus not visualizing dicts')
        return
    fig, ax = plt.subplots(1, len(solvers), figsize=(7*len(solvers), 7))
    for i, (k, v) in enumerate(solvers.items()):
        tiled = su.tiledict(v.getdict().squeeze())
        if tiled.ndim == 2:
            ax[i].imshow(tiled, cmap='gray')
        else:
            ax[i].imshow(tiled)
        ax[i].set_title(k)
        if cfg.SNAPSHOT:
            fig0, ax0 = plt.subplots()
            if tiled.ndim == 2:
                ax0.imshow(tiled, cmap='gray')
            else:
                ax0.imshow(tiled)
            fig0.savefig(os.path.join(cfg.OUTPUT_PATH, k, 'dict.pdf'),
                         bbox_inches='tight')
            plt.close(fig0)
    plt.show()


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
        '{:s}.{:%Y-%m-%d_%H-%M-%S}.{:s}.log'.format(
            cfg.NAME,
            datetime.datetime.now(),
            'train_test' if args.run_test else 'train'
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

    solvers = train_models(defs['TRAIN'])
    if cfg.SNAPSHOT:
        time_stats = {
            k: snapshot_solver_stats(v, os.path.join(cfg.OUTPUT_PATH, k))
            for k, v in solvers.items()
        }
    visualize_dicts(solvers)
    if args.run_test:
        # not implemented for now
        pass


if __name__ == "__main__":
    main()