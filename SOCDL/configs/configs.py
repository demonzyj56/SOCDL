"""Config system mimicing py-faster-rcnn and Detectron.
Since we need to do cross validation and grid search, the configs are
different for different experiments which are run simultaneously.

The cfg here is assumed to be a `global` one.
"""
import copy
import logging
import os
from ast import literal_eval
from future.utils import string_types
import numpy as np
import yaml

from .attr_dict import AttrDict, to_plain_dict

logger = logging.getLogger(__name__)

__C = AttrDict()
cfg = __C

#-----------------------------#
# Common options
#-----------------------------#

# The name of the experiment.
__C.NAME = 'default'

# Rng to set. Negative values means randomly set.
__C.RNG_SEED = 9527

# The root path of cocoa.
__C.ROOT_PATH = os.path.join(os.path.dirname(__file__), '..', '..')

# Default cache path.
__C.CACHE_PATH = os.path.join(__C.ROOT_PATH, '.data')

# Output directory for experiments
__C.OUTPUT_PATH = os.path.join(__C.ROOT_PATH, '.default')

# Default data type.  Currently only float32 is supported.
__C.DATA_TYPE = 'float32'

# Whether to print more verbose information. Note: verbose information is NOT
# logged.
__C.VERBOSE = True

# Whether to snapshot
__C.SNAPSHOT = True

# Epochs to run for training
__C.EPOCHS = 50

# Batch size for training
__C.BATCH_SIZE = 32

# Height and width for dictionaries
__C.PATCH_SIZE = 8

# Number of atoms
__C.NUM_ATOMS = 100

# Lambda
__C.LAMBDA = 1.

# Number of processes to invoke for testing.
__C.NUM_PROCESSES = os.cpu_count()

# Whether to use GPU to test
__C.GPU_TEST = False

#-----------------------------#
# Dataset options
#-----------------------------#

__C.DATASET = AttrDict()

# Which dataset to use
__C.DATASET.NAME = 'fruit'

# Whether to use grayscale images
__C.DATASET.GRAY = False

# A unified size for all images
# This is not valid for some datasets (e.g. fruit/city/cifar10).
__C.DATASET.SIZE = (256, 256)

# Pad the input blob with zeros
__C.DATASET.PAD_BOUNDARY = True

# Whether to apply tikhonov filter before solving CSC
__C.DATASET.TIKHONOV = True

# Read a list of images as dataset.  This is only valid when the name of the
# dataset is `images`.
__C.DATASET.IMAGE_NAMES = []


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    # a list or tuple should be kept for cross validation
    elif isinstance(value_b, string_types) and not isinstance(value_a, (list, tuple)):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    # int->float: this should mostly be acceptable.
    elif isinstance(value_a, int) and isinstance(value_b, float):
        value_a = float(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a


def parse_params_from_file(cfg_file):
    """From a cfg_file, returns the params for cross validation.
    The cfg_file is different with ordinary yaml files used by Detectron.
    In specific, the fields that we want to cross validate is a tuple
    containing desired parameters, instead of a single str/numeric entry.

    For example, in original config file we have e.g. svm.C=100., while
    here in cfg_file we have svm.C=[1., 10., 100.].  Note that each entry
    of the list should be the same type as that in cfg.

    For mergable fields (i.e. single entry), merge to global cfg.
    """
    # The original config version first convert the root node to AttrDict,
    # but leave the leaf nodes to _decode_cfg_value.
    # Here we instead use ordinary dict and build/check an AttrDict
    # recursively.
    def _build_attr_dict_from_dict(d):
        """For a given dict, recursively build the AttrDict."""
        root = AttrDict()
        for k, v in d.items():
            if isinstance(v, dict):
                root[k] = _build_attr_dict_from_dict(v)
            else:
                root[k] = v
        return root

    def _validate_structure(a, b, stack=None):
        """Validate that a has the same structure as b. Returns a pruned
        version of a where the coerced entries are removed."""
        assert isinstance(a, AttrDict) and isinstance(b, AttrDict)
        d = AttrDict()
        for k, v in a.items():
            # check key
            full_key = '.'.join(stack) + '.' + k if stack is not None else k
            if k not in b:
                raise KeyError('Non-existent config key: {}'.format(full_key))
            # check value
            if isinstance(v, AttrDict):
                # v is AttrDict: validate recursively
                stack_push = [k] if stack is None else stack + [k]
                d[k] = _validate_structure(v, b[k], stack=stack_push)
            else:
                try:
                    value = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)
                    b[k] = value
                except ValueError:
                    d[k] = [_check_and_coerce_cfg_value_type(vv, b[k], k, full_key) for vv in v]
                except:
                    raise

        return d

    attr_cfg = _build_attr_dict_from_dict(yaml.load(open(cfg_file, 'r')))
    cv_cfg = _validate_structure(attr_cfg, cfg)

    return to_plain_dict(cv_cfg, sep='.')
