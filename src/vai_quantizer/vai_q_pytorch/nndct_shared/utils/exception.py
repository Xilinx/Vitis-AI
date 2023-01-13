

#
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
#

from nndct_shared.base import NNDCT_KEYS, GLOBAL_MAP
from typing import Sequence, Union, Optional
class ReachTheBaseExp(Exception):

  def __init__(self, cls):
    err = "Reach the Base of {}, no more items return".format(cls)
    Exception.__init__(self, err)

class RebuildMismatchError(Exception):

  def __init__(self, file_name=None):
    file_name = file_name or GLOBAL_MAP.get_ele(
        NNDCT_KEYS.MODIFIER).nndct_prefix + '.py'
    Exception.__init__(
        self,
        "The rebuilt graph mismatch with original graph, please manually modify '{}' and run again"
        .format(file_name))

class DeprecatedAPIError(Exception):

  def __init__(self, api_name: str, class_type=None):
    if class_type and isinstance(class_type, str):
      err = 'The api "{api_name}" in {class_type} is deprecated'.format(
          api_name=api_name, class_type=class_type)
    else:
      err = 'The api "{api_name}" is deprecated'.format(api_name=api_name)
    Exception.__init__(self, err)

class AddXopError(Exception):

  def __init__(self, op_name: Union[Sequence, str], op_type: Optional[str] = None, msg: Optional[str] = None):
    if isinstance(op_name, str):
      Exception.__init__(
          self, "Failed to add op(name:{}, type:{}) in xGraph.: '{}'".format(
              op_name, op_type, msg))
    else:
       Exception.__init__(
          self, f"Please support ops({op_name}) in xGraph.")

class ExportXmodelError(Exception):

  def __init__(self, graph_name):
    Exception.__init__(
        self, "Failed to convert nndct graph({}) to xmodel!".format(graph_name))

class DefineOptionError(Exception):

  def __init__(self, option_name, msg):
    Exception.__init__(
        self, "Option '{option_name}' initiate failed '{detail_msg}'!".format(
            option_name=option_name, detail_msg=msg))

class DataXopError(Exception):

  def __init__(self, op_name: Union[Sequence, str], shape):
      Exception.__init__(
          self, "Failed to add data op(name:{}, shape:{}) in xGraph.".format(
              op_name, shape))
