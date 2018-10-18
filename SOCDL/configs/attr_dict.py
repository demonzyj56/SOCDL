# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""A simple attribute dictionary used for representing configuration options.
Rename collections.py -> attr_dict.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class AttrDict(dict):

    IMMUTABLE = '__immutable__'

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__[AttrDict.IMMUTABLE] = False

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not self.__dict__[AttrDict.IMMUTABLE]:
            if name in self.__dict__:
                self.__dict__[name] = value
            else:
                self[name] = value
        else:
            raise AttributeError(
                'Attempted to set "{}" to "{}", but AttrDict is immutable'.
                format(name, value)
            )

    def immutable(self, is_immutable):
        """Set immutability to is_immutable and recursively apply the setting
        to all nested AttrDicts.
        """
        self.__dict__[AttrDict.IMMUTABLE] = is_immutable
        # Recursively set immutable state
        for v in self.__dict__.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)
        for v in self.values():
            if isinstance(v, AttrDict):
                v.immutable(is_immutable)

    def is_immutable(self):
        return self.__dict__[AttrDict.IMMUTABLE]


def to_plain_dict(attr_dict, sep=None):
    """
    Converts an AttrDict object to a `plain_dict`.
    The term `plain_dict` means that there is no nested dict inside the
    returned dict.  For nested field in input attr_dict, the attributes
    are joined using a separator.  If `sep` is None, then the default
    separator is `__`.
    For example, cfg.svm.C=100.0 will be converted to
    cfg_plain['svm__C'] = 100.0.
    """
    assert isinstance(attr_dict, AttrDict)
    if sep is None:
        sep = '__'
    out = dict()
    for k, v in attr_dict.items():
        if isinstance(v, AttrDict):
            # recursively get the dict and join
            vdict = to_plain_dict(v, sep=sep)
            for kk, vv in vdict.items():
                out.update({sep.join([k, kk]): vv})
        else:
            out[k] = v

    return out


def from_plain_dict(plain_dict, sep=None):
    """Converts a converted plain_dict back to AttrDict.
    If `sep` is None, then the default separator is `__`.
    """
    assert isinstance(plain_dict, dict)
    assert all([not isinstance(v, dict) for v in plain_dict.values()])
    if sep is None:
        sep = '__'
    attr_dict = AttrDict()
    for k, v in plain_dict.items():
        all_keys = k.split(sep)
        current = attr_dict
        for ak in all_keys:
            if ak == all_keys[-1]:  # the last one
                current[ak] = v
            else:
                if not hasattr(current, ak):
                    current[ak] = AttrDict()
                current = current[ak]

    return attr_dict
