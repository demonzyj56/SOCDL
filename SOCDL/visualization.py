"""Visualize final results."""
import logging
import os
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError:
    plt = None
from .configs import cfg as _cfg

logger = logging.getLogger(__name__)


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

    if _cfg.SNAPSHOT:
        plt.savefig(os.path.join(_cfg.OUTPUT_PATH, 'statistics.pdf'),
                    bbox_inches='tight')
    plt.show()