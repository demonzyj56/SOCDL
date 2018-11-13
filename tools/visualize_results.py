#!/usr/bin/env python
"""Visualizations of final results."""
import argparse
import copy
import logging
import pickle
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
    parser.add_argument('--no_show', action='store_true',
                        help='If activated, then the figure is not shown')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='Command line arguments')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def plot_obj_vs_time(fncs, time_stats, class_legend=None, show=True):
    plt.clf()
    for k, v in time_stats.items():
        label = class_legend.get(k, k) if class_legend is not None else k
        plt.plot(v, fncs[k], label=label)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Test set objective')
    if cfg.SNAPSHOT:
        path = os.path.join(cfg.OUTPUT_PATH, 'obj_vs_time.pdf')
        logger.info('Saving obj1 to %s', path)
        plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show()


def plot_obj_vs_iteration(fncs, class_legend=None, show=True):
    plt.clf()
    for k, v in fncs.items():
        label = class_legend.get(k, k) if class_legend is not None else k
        plt.plot(fncs[k], label=label)
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Test set objective')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    if cfg.SNAPSHOT:
        path = os.path.join(cfg.OUTPUT_PATH, 'obj_vs_iteration.pdf')
        logger.info('Saving obj2 to %s', path)
        plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show()


def plot_psnr_vs_time(psnrs, time_stats, class_legend=None, show=True):
    plt.clf()
    for k, v in time_stats.items():
        label = class_legend.get(k, k) if class_legend is not None else k
        plt.plot(v, psnrs[k], label=label)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('PSNR (dB)')
    if cfg.SNAPSHOT:
        path = os.path.join(cfg.OUTPUT_PATH, 'psnr_vs_time.pdf')
        logger.info('Saving obj3 to %s', path)
        plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show()


class Plotter(object):
    """Plotting object."""

    def __init__(self, runner, class_legend=None, show=True):
        self.time_stats = copy.deepcopy(runner.time_stats)
        self.fncs, self.psnrs = {}, {}
        for k, v in runner.results.items():
            fnc, psnr = list(zip(*v))
            self.fncs.update({k: copy.deepcopy(fnc)})
            self.psnrs.update({k: copy.deepcopy(psnr)})
        self.class_legend = class_legend
        self.show = show

    def plot_obj_vs_time(self):
        plot_obj_vs_time(self.fncs, self.time_stats,
                         self.class_legend, self.show)

    def plot_obj_vs_iteration(self):
        plot_obj_vs_iteration(self.fncs, self.class_legend, self.show)

    def plot_psnr_vs_time(self):
        plot_psnr_vs_time(self.psnrs, self.time_stats,
                          self.class_legend, self.show)


def main():
    """Main entry."""
    args = parse_args()
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert os.path.exists(cfg.OUTPUT_PATH)
    setup_logging()
    logger.info('Using matplotlibrc file from %s', mpl.matplotlib_fname())
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
    # pack results
    runner.export_data()
    plotter = Plotter(runner, show=(not args.no_show))
    plotter.plot_obj_vs_time()
    plotter.plot_obj_vs_iteration()
    plotter.plot_psnr_vs_time()


if __name__ == '__main__':
    main()
