#!/usr/bin/env python
"""Visualizations of final results."""
import argparse
import logging
import pickle
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# add SOCDL working directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from SOCDL.configs.configs import cfg, merge_cfg_from_file, merge_cfg_from_list
from SOCDL.utils import setup_logging
from SOCDL.builder import collect_time_stats

logger = logging.getLogger(__name__)


def parse_args():
    """Argument parser."""
    parser = argparse.ArgumentParser(description='Result visualization.')
    parser.add_argument('--cfg', dest='cfg_file', default=None, type=str,
                        help='cfg file to parse options from')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Command line arguments')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def plot_statistics(results, time_stats, class_legend=None):
    """Plot obtained statistics."""
    fncs, psnrs = {}, {}
    for k, v in results.items():
        fnc, psnr = list(zip(*v))
        fncs.update({k: fnc})
        psnrs.update({k: psnr})

    # create fig
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))

    for k, v in time_stats.items():
        if class_legend is not None:
            label = class_legend.get(k, k)
        else:
            label = k
        ax[0].plot(v, fncs[k], label=label, linewidth=2.5)
        ax[1].plot(fncs[k], label=label, linewidth=2.5)
        ax[2].plot(v, psnrs[k], label=label, linewidth=2.5)
    for xx in ax:
        xx.legend()
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Test set objective')
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Test set objective')
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel('PSNR (dB)')

    if cfg.SNAPSHOT:
        plt.savefig(os.path.join(cfg.OUTPUT_PATH, 'statistics.pdf'),
                    bbox_inches='tight')
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
    with open(os.path.join(cfg.OUTPUT_PATH, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)
    # collect time stats
    time_stats = {k: collect_time_stats(k) for k in results.keys()}
    plot_statistics(results, time_stats)


if __name__ == '__main__':
    main()