# Copyright 2022 Xilinx Inc.
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

from enum import auto, unique, Enum

class AutoName(Enum):
  def _generate_next_value_(name, start, count, last_values):
    return name.lower()

@unique
class DeviceType(AutoName):
  CPU = auto()
  DPU = auto()

class DeviceInfo(object):
  def __init__(self, device_type):
    assert isinstance(device_type, DeviceType)
    self._type = device_type
    self._device_partition_check_msg = None

  def get_device_type(self):
    return self._type

  def set_filter_message(self, msg):
    self._device_partition_check_msg = msg
  
  def get_filter_message(self):
    return self._device_partition_check_msg

  def clear_filter_message(self):
    self._device_partition_check_msg = None
