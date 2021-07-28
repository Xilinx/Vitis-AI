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
# ==============================================================================
"""Common Utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json


class VAILogger(object):
  """Vitis AI logger.

  Default log level is 0.

  log level:
    -1: print VAI_INFO, VAI_WARNING, VAI_DEBUG
     0: print VAI_INFO, VAI_WARNING
     1: print WARNING
  """
  _default_log_level = 0

  @classmethod
  def get_log_level(cls):
    """Get the log level from environment, default level is 0."""
    if 'VAI_LOG_LEVEL' in os.environ:
      return int(os.environ.get('VAI_LOG_LEVEL'))
    else:
      return cls._default_log_level

  @classmethod
  def set_log_level(cls, new_level):
    """Set the log level environment."""
    os.environ.set('VAI_LOG_LEVEL', new_level)

  @staticmethod
  def debug(msg):
    """Print VAI DEBUG messages."""
    if VAILogger.get_log_level() <= -1:
      print('[VAI DEBUG] ' + msg)

  @staticmethod
  def info(msg):
    """Print VAI INFO messages."""
    if VAILogger.get_log_level() <= 0:
      print('[VAI INFO] ' + msg)

  @staticmethod
  def warning(msg):
    """Print VAI WARNING messages."""
    if VAILogger.get_log_level() <= 1:
      print('[VAI WARNING] ' + msg)

  @staticmethod
  def error(msg, err_type=ValueError):
    """Print VAI ERROR messages."""
    raise err_type('[VAI ERROR] ' + msg)

  @staticmethod
  def debug_enabled():
    return VAILogger.get_log_level() <= -1


logger = VAILogger


def load_json(json_file):
  """Load json file."""
  with open(json_file, 'r') as f:
    try:
      data = json.loads(f.read())
    except Exception as e:
      logger.error(
          'Fail to load the json file `{}`, please check the format. \nError: {}'
          .format(json_file, e))
  return data
