#!/usr/bin/env python
"""Model testing routines."""
import argparse
import datetime
import logging
import pickle
import pprint
import re
import os
import sys
import time
import yaml
import pyfftw  # pylint: disable=unused-import
import numpy as np
import torch

# add SOCDL working directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from SOCDL.configs.configs import cfg, merge_cfg_from_file, merge_cfg_from_list
from SOCDL.builder import get_loader
from SOCDL.runtest import map_cbpdn_dicts
from SOCDL.utils import setup_logging
import visualize_results

logger = logging.getLogger(__name__)


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Testing the given models.')
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


def collect_dictionaries(key):
    """Collect dictionaries for a learned and cached solver."""
    base_dir = os.path.join(cfg.OUTPUT_PATH, key)
    try:
        files = os.listdir(base_dir)
    except:
        logger.info('Folder %s does not exists, no dict collected', base_dir)
        return
    pattern = re.compile(r'[0-9]+\.npy')
    files = [f for f in files if pattern.match(f) is not None]
    # sort paths according to index
    files = sorted(files, key=lambda t: int(t.split('.')[-2]))
    files = [os.path.join(base_dir, f) for f in files]
    assert all([os.path.exists(f) for f in files])
    dicts = [np.load(f) for f in files]
    logger.info('Collected %d dictionaries from %s', len(dicts), base_dir)
    return dicts


def eval_models(dicts, defs):
    """Run tests over given dictionaries."""
    loader = get_loader(train=False)
    results = {k: None for k in dicts.keys()}
    for e, (sl, sh) in enumerate(loader):
        for k, ds in dicts.items():
            logger.info(
                'Running CBPDN over %d dictionaries for %s on epoch %d/%d',
                len(ds), k, e+1, cfg.EPOCHS
            )
            tic = time.time()
            r = map_cbpdn_dicts(ds, sl, sh, cfg.LAMBDA, sl+sh, defs['ConvBPDN'])
            logger.info('Done in %.3fs.', time.time()-tic)
            logger.info('(functional value, PSNR):')
            logger.info(pprint.pformat(r))
            if results[k] is None:
                results[k] = r
            else:
                results[k] = [(fnc1+fnc2, psnr1+psnr2) for
                              (fnc1, psnr1), (fnc2, psnr2) in zip(results[k], r)]
    # PSNR should be averaged, assuming all epochs have same size
    results = {k: [(fnc, psnr/loader.epochs) for fnc, psnr in v]
               for k, v in results.items()}
    if cfg.SNAPSHOT:
        with open(os.path.join(cfg.OUTPUT_PATH, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        logger.info('Final test results for %s has been saved to %s',
                    cfg.NAME, os.path.join(cfg.OUTPUT_PATH, 'results.pkl'))

    return results


def main():
    """Main entry."""
    args = parse_args()
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert os.path.exists(cfg.OUTPUT_PATH), \
        'Output path {} does not exist'.format(cfg.OUTPUT_PATH)
    log_name = os.path.join(
        cfg.OUTPUT_PATH,
        '{:s}.{:%Y-%m-%d_%H-%M-%S}.test.log'.format(
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
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))
    if cfg.RNG_SEED >= 0:
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed_all(cfg.RNG_SEED)
    with open(args.def_file, 'r') as f:
        defs = yaml.load(f)
    solver_names = defs['TRAIN'].keys()
    logger.info('Testing solvers:')
    logger.info(solver_names)

    dicts = {k: collect_dictionaries(k) for k in solver_names}
    results = eval_models(dicts, defs['TEST'])
    time_stats = {
        k: visualize_results.collect_time_stats(k) for k in solver_names
    }
    visualize_results.plot_statistics(results, time_stats)


if __name__ == '__main__':
    main()