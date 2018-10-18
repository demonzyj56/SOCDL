"""Build the corresponding learners given the configs."""
import os
import numpy as np
from sporco.dictlrn.cbpdndl import ConvBPDNDictLearn
from .configs import cfg as _cfg
from .impl import *
from .datasets import *
from .datasets.transforms import default_transform


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

    def _callback(d):
        """Snapshot dictionaries for every iteration."""
        if _cfg.SNAPSHOT:
            _D = d.getdict().squeeze()
            np.save(os.path.join(
                path, '{}.{}.npy'.format(_cfg.DATASET.NAME, d.j)), _D)
        return 0

    opt['Callback'] = _callback
    solver = ConvBPDNDictLearn(D0, samples, _cfg.LAMBDA, opt=opt,
                               xmethod='admm', dmethod='cns')
    return {'ConvBPDNDictLearn': solver}


def get_loader(train=True):
    """Get loader, with default arguments."""
    transform = lambda blob: default_transform(blob, _cfg.PATCH_SIZE//2,
                                               _cfg.DATASET.TIKHONOV)
    args = dict(
        epochs=_cfg.EPOCHS,
        batch_size=_cfg.BATCH_SIZE,
        train=train,
        dtype=np.float32,
        scaled=True,
        gray=_cfg.DATASET.GRAY,
        transform=transform
    )
    name = _cfg.DATASET.NAME
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
        loader = VOC07Loader(root=_cfg.CACHE_PATH, dsize=_cfg.DATASET.SIZE,
                             **args)
    elif name == '17flowers':
        loader = VGG17FlowersLoader(root=_cfg.CACHE_PATH,
                                    dsize=_cfg.DATASET.SIZE, **args)
    elif name == '102flowers':
        loader = VGG102FlowersLoader(root=_cfg.CACHE_PATH,
                                     dsize=_cfg.DATASET.SIZE, **args)
    elif name == 'images':
        args.pop('train')
        loader = ImageLoader(names=_cfg.DATASET.IMAGE_NAMES,
                             dsize=_cfg.DATASET.SIZE, **args)
    else:
        raise KeyError('Unknown loader name: {}'.format(name))
    return loader

