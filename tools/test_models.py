#!/usr/bin/env python
"""Model testing routines."""
import argparse
import datetime
import logging
import pickle
import pprint
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
from SOCDL.builder import get_loader, collect_time_stats, collect_dictionaries
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


def eval_models(dicts, defs):
    """Run tests over given dictionaries."""
    loader = get_loader(train=False)
    results = {k: None for k in dicts.keys()}
    for e, (sl, sh) in enumerate(loader):
        for k, ds in dicts.items():
            logger.info(
                'Running CBPDN over %d dictionaries for %s on epoch %d/%d',
                len(ds), k, e+1, loader.epochs
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


class GenericTestRunner(object):
    """Generic test runner that works for most of cases."""

    def __init__(self, defs):
        self.defs = defs
        self.solver_names = defs['TRAIN'].keys()
        self.time_stats = {k: collect_time_stats(k) for k in self.solver_names}
        self.dicts = {k: collect_dictionaries(k) for k in self.solver_names}
        self.results = None
        self.max_num_dict = cfg.TEST.MAX_NUM_DICT
        for k in self.solver_names:
            assert len(self.time_stats[k]) == len(self.dicts[k])
            # Optionally drop first k samples.
            if cfg.TEST.DROP_FIRST_K > 0 and \
                    cfg.TEST.DROP_FIRST_K < len(self.dicts[k]):
                logger.info('Dropping the first %d of %d dictionaries from %s',
                            cfg.TEST.DROP_FIRST_K, len(self.dicts[k]), k)
                self.time_stats[k] = [
                    t - self.time_stats[k][cfg.TEST.DROP_FIRST_K-1]
                    for i, t in enumerate(self.time_stats[k])
                    if i >= cfg.TEST.DROP_FIRST_K
                ]
                self.dicts[k] = self.dicts[k][cfg.TEST.DROP_FIRST_K:]
            # sample max_num_dict from all dicts
            if self.max_num_dict > 0 and self.max_num_dict < len(self.dicts[k]):
                logger.info('Uniformly sampled %d/%d dictionaries from %s',
                            self.max_num_dict, len(self.dicts[k]), k)
                vt = self.time_stats[k]
                self.time_stats[k] = [
                    vt[int(np.ceil(i*len(vt)/self.max_num_dict))]
                    for i in range(self.max_num_dict)
                ]
                vd = self.dicts[k]
                self.dicts[k] = [
                    vd[int(np.ceil(i*len(vd)/self.max_num_dict))]
                    for i in range(self.max_num_dict)
                ]

    def run(self):
        """The the test solver."""
        logger.info('Testing sovlers:')
        logger.info(self.solver_names)
        if 'DUPLICATED' in self.defs['TEST']:
            names = self.defs['TEST']['DUPLICATED'].keys()
            dicts = {k: v for k, v in self.dicts.items() if k not in names}
            logger.info('Skip testing %s to avoid duplicated computations',
                        ', '.join(names))
        else:
            names = []
            dicts = self.dicts
        self.results = eval_models(dicts, self.defs['TEST'])
        for name in names:
            target = self.defs['TEST']['DUPLICATED'][name]
            self.results.update({name: self.results[target]})

        return self.results

    def plot_statistics(self):
        """Plot the computed statistics."""
        visualize_results.plot_statistics(self.results, self.time_stats)


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
    runner = GenericTestRunner(defs)
    runner.run()
    runner.plot_statistics()


if __name__ == '__main__':
    main()