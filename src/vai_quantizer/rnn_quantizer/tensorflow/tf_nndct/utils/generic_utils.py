

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

from google.protobuf import text_format

def to_list(x):
  """Normalizes a list/tuple to a list.

  If a tensor is passed, we return
  a list of size 1 containing the tensor.

  Arguments:
      x: target object to be normalized.

  Returns:
      A list.
  """
  if isinstance(x, (list, tuple)):
    return list(x)
  return [x]

def get_temp_directory():
  return os.environ.get("VAI_TEMP_DIRECTORY", tempfile.mkdtemp())

def delete_directory(path):
  if os.path.exists(path):
    shutil.rmtree(path)

def mkdir_if_not_exist(x):
  if not x or os.path.isdir(x):
    return
  os.mkdir(x)
  if not os.path.isdir(x):
    raise RuntimeError("Failed to create dir %r" % x)

def write_proto(path, message, as_text=False):
  dir_name = os.path.dirname(path)
  mkdir_if_not_exist(dir_name)
  if dir_name:
    os.makedirs(dir_name, exist_ok=True)
  if as_text:
    with open(path, "w") as f:
      f.write(text_format.MessageToString(message))
  else:
    with open(path, "wb") as f:
      f.write(message.SerializeToString())

def write_text_proto(path, message):
  write_proto(path, message, as_text=True)

def write_binary_proto(path, message):
  write_proto(path, message, as_text=False)

def is_list_or_tuple(obj):
  return isinstance(obj, (list, tuple))
