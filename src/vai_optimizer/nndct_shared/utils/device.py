
from enum import auto, unique
from .common import AutoName


@unique
class DeviceType(AutoName):
  CPU = auto()
  DPU = auto()
  USER = auto()

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
