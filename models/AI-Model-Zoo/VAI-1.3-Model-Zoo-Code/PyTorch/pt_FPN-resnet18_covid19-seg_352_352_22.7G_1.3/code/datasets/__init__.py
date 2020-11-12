# Copyright 2019 Xilinx Inc.
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

# Copyright 2019 Xilinx Inc.
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

import warnings
from torchvision.datasets import *
from code.datasets.base import *
from code.datasets.cityscapes import CitySegmentation
#from .cityscapes_bdd import CitySegmentation

class EncodingDeprecationWarning(DeprecationWarning):
    pass

warnings.simplefilter('once', EncodingDeprecationWarning)

datasets = {
    'citys': CitySegmentation,
}

acronyms = {
    'citys': 'citys',
    'cityscapes': 'citys'
}

def get_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)

def _make_deprecate(meth, old_name):
    new_name = meth.__name__

    def deprecated_init(*args, **kwargs):
        return meth(*args, **kwargs)

    deprecated_init.__doc__ = r"""
    {old_name}(...)
    .. warning::
        This method is now deprecated in favor of :func:`torch.nn.init.{new_name}`.
    See :func:`~torch.nn.init.{new_name}` for details.""".format(
        old_name=old_name, new_name=new_name)
    deprecated_init.__name__ = old_name
    return deprecated_init

get_segmentation_dataset = _make_deprecate(get_dataset, 'get_segmentation_dataset')
