#!/usr/bin/env python
"""Visualizations of final results."""
import argparse
import logging
import pickle
import os
import sys
import matplotlib.pyplot as plt
import yaml

# add SOCDL working directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from SOCDL.configs.configs import cfg, merge_cfg_from_file, merge_cfg_from_list
from SOCDL.utils import setup_logging
from test_models import GenericTestRunner

logger = logging.getLogger(__name__)


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Result visualization.')
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


def plot_statistics(results, time_stats, class_legend=None):
    """Plot the statistics."""
    fncs, psnrs = {}, {}
    for k, v in results.items():
        fnc, psnr = list(zip(*v))
        fncs.update({k: fnc})
        psnrs.update({k: psnr})

    for k, v in time_stats.items():
        label = class_legend.get(k, k) if class_legend is not None else k
        plt.plot(v, fncs[k], label=label, linewidth=2.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Test set objective')
    plt.legend()
    plt.show()


def main():
    """Main entry."""
    args = parse_args()
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert os.path.exists(cfg.OUTPUT_PATH)
    setup_logging()
    # collect results
    if os.path.exists(os.path.join(cfg.OUTPUT_PATH, 'runner.pkl')):
        logger.info('Loading runner from %s',
                    os.path.join(cfg.OUTPUT_PATH, 'runner.pkl'))
        with open(os.path.join(cfg.OUTPUT_PATH, 'runner.pkl'), 'rb') as f:
            runner = pickle.load(f)
    else:
        assert args.def_file is not None and os.path.exists(args.def_file)
        with open(args.def_file, 'r') as f:
            defs = yaml.load(f)
        runner = GenericTestRunner(defs)
        runner.load_results()
    runner.plot_statistics()


if __name__ == '__main__':
    main()
