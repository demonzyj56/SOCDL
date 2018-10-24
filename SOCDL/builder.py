"""Build the corresponding learners given the configs."""
import functools
import logging
import re
import os
import pickle
import numpy as np
import sporco.util as su
from sporco.dictlrn.cbpdndl import ConvBPDNDictLearn
from .configs import cfg as _cfg
from .impl import *
from .datasets import *
from .datasets.transforms import default_transform

logger = logging.getLogger(__name__)
pattern = re.compile(r'[0-9]+\.npy')


def snapshot_solver_dict(solver, path, cur_cnt=None):
    """Saving solver's current dict to path.  This requires that the solver
    has method `getdict()`."""
    stype = type(solver).__name__
    if _cfg.SNAPSHOT:
        assert hasattr(solver, 'getdict'), \
            'Cannot snapshot {} since it does not ' \
            'have `getdict` method.'.format(stype)
        # parse filename
        if cur_cnt is not None:
            cnt = cur_cnt
        elif hasattr(solver, 'j'):
            cnt = solver.j
        else:
            logger.warning('%s does not have iteration counter, try parsing'
                           'from path %s', stype, path)
            cnt = max([int(f.split('.')[-2]) for f in os.listdir(path)
                       if pattern.match(f) is not None])

        _D = solver.getdict().squeeze()
        np.save(os.path.join(path, '{}.npy'.format(cnt)), _D)
    return 0


def snapshot_solver_stats(solver, path):
    """Save solver's running time statistics (and others) to path."""
    if _cfg.SNAPSHOT:
        stats_arr = su.ntpl2array(solver.getitstat())
        np.save(os.path.join(path, 'stats.npy'), stats_arr)
        time_stats = {'Time': solver.getitstat().Time}
        with open(os.path.join(path, 'time_stats.pkl'), 'wb') as f:
            pickle.dump(time_stats, f)
        return time_stats
    return None


def get_online_solvers(cfg, D0, init_sample):
    """Construct online solvers from given cfg."""
    solvers = {}
    for k, v in cfg.items():
        solver_class = globals()[k.split('-')[0]]
        opt = solver_class.Options(v)
        path = os.path.join(_cfg.OUTPUT_PATH, k)
        if not os.path.exists(path):
            os.makedirs(path)
        solvers[k] = solver_class(D0, init_sample, _cfg.LAMBDA, opt=opt)
    return solvers


def get_batch_solver(cfg, D0, samples):
    """Construct batch solver from given cfg."""
    opt = ConvBPDNDictLearn.Options(cfg, xmethod='admm', dmethod='cns')
    path = os.path.join(_cfg.OUTPUT_PATH, 'ConvBPDNDictLearn')
    if not os.path.exists(path):
        os.makedirs(path)

    opt['Callback'] = functools.partial(snapshot_solver_dict,
                                        path=path, cur_cnt=None)
    solver = ConvBPDNDictLearn(D0, samples, _cfg.LAMBDA, opt=opt,
                               xmethod='admm', dmethod='cns')
    return {'ConvBPDNDictLearn': solver}


def get_loader(train=True):
    """Get loader, with default arguments."""
    dcfg = _cfg.TRAIN.DATASET if train else _cfg.TEST.DATASET
    if dcfg.PAD_BOUNDARY:
        pad_size = _cfg.PATCH_SIZE // 2
    else:
        pad_size = None
    transform = lambda blob: default_transform(blob, pad_size, dcfg.TIKHONOV)
    args = dict(
        epochs=_cfg.TRAIN.EPOCHS if train else _cfg.TEST.EPOCHS,
        batch_size=_cfg.TRAIN.BATCH_SIZE if train else _cfg.TEST.BATCH_SIZE,
        train=train,
        dtype=np.float32,
        scaled=True,
        gray=dcfg.GRAY,
        transform=transform
    )
    name = dcfg.NAME
    if name == 'cifar10':
        loader = CIFAR10Loader(root=_cfg.CACHE_PATH, **args)
    elif name == 'cifar100':
        loader = CIFAR100Loader(root=_cfg.CACHE_PATH, **args)
    elif name == 'fruit':
        loader = FruitLoader(use_processed=False, **args)
    elif name == 'city':
        loader = CityLoader(use_processed=False, **args)
    elif name == 'fruit.ocsc':
        loader = FruitLoader(use_processed=True, **args)
    elif name == 'city.ocsc':
        loader = CityLoader(use_processed=True, **args)
    elif name == 'voc07':
        loader = VOC07Loader(root=_cfg.CACHE_PATH, dsize=dcfg.SIZE, **args)
    elif name == '17flowers':
        loader = VGG17FlowersLoader(root=_cfg.CACHE_PATH,
                                    dsize=dcfg.SIZE, **args)
    elif name == '102flowers':
        loader = VGG102FlowersLoader(root=_cfg.CACHE_PATH,
                                     dsize=dcfg.SIZE, **args)
    elif name == 'images':
        args.pop('train')
        loader = ImageLoader(names=dcfg.IMAGE_NAMES, dsize=dcfg.SIZE, **args)
    elif name == 'stl10':
        args.pop('train')
        category = 'train' if train else 'test'
        loader = STL10Loader(root=_cfg.CACHE_PATH, category=category, **args)
    elif name == 'stl10.unlabeled':
        args.pop('train')
        loader = STL10Loader(root=_cfg.CACHE_PATH, category='unlabeled', **args)
    else:
        raise KeyError('Unknown loader name: {}'.format(name))
    return loader


def collect_dictionaries(key):
    """Collect dictionaries for a learned and cached solver."""
    base_dir = os.path.join(_cfg.OUTPUT_PATH, key)
    try:
        files = os.listdir(base_dir)
    except:
        logger.info('Folder %s does not exists, no dict collected', base_dir)
        return
    files = [f for f in files if pattern.match(f) is not None]
    # sort paths according to index
    files = sorted(files, key=lambda t: int(t.split('.')[-2]))
    files = [os.path.join(base_dir, f) for f in files]
    assert all([os.path.exists(f) for f in files])
    dicts = [np.load(f) for f in files]
    logger.info('Collected %d dictionaries from %s', len(dicts), base_dir)
    return dicts


def collect_time_stats(key):
    """Collect time statistics for a learned and cached solver."""
    path = os.path.join(_cfg.OUTPUT_PATH, key)
    if os.path.exists(os.path.join(path, 'time_stats.pkl')):
        with open(os.path.join(path, 'time_stats.pkl'), 'rb') as f:
            time_stats = pickle.load(f)
        return time_stats['Time']
    else:
        assert os.path.exists(os.path.join(path, 'stats.npy'))
        s = np.load(os.path.join(path, 'stats.npy'))
        return s[0][s[1].index('Time')]