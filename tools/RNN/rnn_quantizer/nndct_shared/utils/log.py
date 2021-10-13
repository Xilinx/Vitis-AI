

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

import sys
import os
import logging
import io

from nndct_shared.base import NNDCT_KEYS, GLOBAL_MAP, NNDCT_DEBUG_LVL

def obj_to_str(obj):
  if isinstance(obj, list):
    string = '\n'.join(["{}".format(n) for n in obj])
  elif isinstance(obj, dict):
    string = '\n'.join(["{} : {}".format(k, v) for k, v in obj.items()])
  elif isinstance(obj, str):
    string = obj
  else:
    raise Exception("nndct_details_debug only support list and dictionary")
  return string

def nndct_info_print(string):
  logger = GLOBAL_MAP.get_ele(NNDCT_KEYS.LOGGER)
  if logger:
    logger.info("[NNDCT_INFO] {}".format(string))
  else:
    print("[NNDCT_INFO] {}".format(string))

def nndct_warn_print(string):
  if True == GLOBAL_MAP.get_ele(NNDCT_KEYS.WARN_FLAG):
    logger = GLOBAL_MAP.get_ele(NNDCT_KEYS.LOGGER)
    if logger:
      logger.warning("[NNDCT_WARN] {}".format(string))
    else:
      print("[NNDCT_WARN] {}".format(string))

def nndct_debug_print(string, title='', level=1):
  if True == GLOBAL_MAP.get_ele(
      NNDCT_KEYS.DEBUG_FLAG) and level <= GLOBAL_MAP.get_ele(
          NNDCT_KEYS.VERBOSE_LEVEL):
    logger = GLOBAL_MAP.get_ele(NNDCT_KEYS.LOGGER)
    if title == 'Start':
      string = "\n********************* <{} : {}> *********************".format(
          title, string)
    elif title == 'End':
      string = "\n********************* <{} : {}> *********************\n".format(
          title, string)
    if logger:
      logger.debug("[NNDCT_DEBUG_Lv_{}] {}".format(level, string))
    else:
      print("[NNDCT_DEBUG_Lv_{}] {}".format(level, string))

def nndct_error_print(string):
  if True == GLOBAL_MAP.get_ele(NNDCT_KEYS.ERROR_FLAG):
    logger = GLOBAL_MAP.get_ele(NNDCT_KEYS.LOGGER)
    if logger:
      logger.error("[NNDCT_ERROR] {}".format(string))
    else:
      print("[NNDCT_ERROR] {}".format(string))
    sys.exit(1)

def nndct_details_debug(obj, title, level=NNDCT_DEBUG_LVL.DETAILS):
  nndct_debug_print(
      "\n********************* <Start : {}> *********************\n{}".format(
          title, obj_to_str(obj)),
      level=level)
  nndct_debug_print(title, title='End', level=level)

#some wrappers
def nndct_info(func):

  def wrapper(*args, **kwargs):
    info_flag = GLOBAL_MAP.get_ele(NNDCT_KEYS.INFO_FLAG)
    if info_flag == True:
      print("[NNDCT_INFO]", end='')
    return func(*args, **kwargs)

  return wrapper

def nndct_warn(func):

  def wrapper(*args, **kwargs):
    warn_flag = GLOBAL_MAP.get_ele(NNDCT_KEYS.WARN_FLAG)
    if warn_flag == True:
      print("[NNDCT_WARN]", end='')
    return func(*args, **kwargs)

  return wrapper

def nndct_debug(func):

  def wrapper(*args, **kwargs):
    debug_flag = GLOBAL_MAP.get_ele(NNDCT_KEYS.DEBUG_FLAG)
    if debug_flag == True:
      print("[NNDCT_DEBUG]", end='')
    return func(*args, **kwargs)

  return wrapper

def nndct_error(func):

  def wrapper(*args, **kwargs):
    error_flag = GLOBAL_MAP.get_ele(NNDCT_KEYS.ERROR_FLAG)
    if error_flag == True:
      print("[NNDCT_ERROR]", end='')
    return func(*args, **kwargs)
    if error_flag == True:
      exit(1)

  return wrapper

def get_nndct_logger(filename='NndctGen_log'):
  log_dir = os.path.dirname(filename)
  if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)
  logger = logging.getLogger(filename.replace("/", 'SPL'))
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter(
      '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
  sh = logging.StreamHandler()
  fh = logging.FileHandler(filename, mode='w', encoding=None, delay=False)
  for h in [sh, fh]:
    h.setLevel(logging.INFO)
    h.setFormatter(formatter)
    logger.addHandler(h)
  return logger

def log_or_print(str, logger=None):
  if logger:
    logger.info(str)
  else:
    print(str)

def get_config_str(obj,
                   title,
                   ignore_prefix=[],
                   ignore_suffix=[],
                   ignore_keys=[]):
  assert hasattr(
      obj, 'default_kwargs'
  ), 'object {} has no default_kwargs, failed to generate configuration string'.format(
      obj)
  config_str = '\n' + ">>    <{}>".format(title) + '\n>>    '
  for key in obj.default_kwargs:
    value = getattr(obj, key, None)
    if value and not any(key.endswith(s) for s in ignore_suffix) and \
        not any(key.startswith(p) for p in ignore_prefix) and \
        key not in ignore_keys:
      if isinstance(value, dict):
        config_str += '\n>>    {}: \n>>        {}'.format(
            key, '\n>>        '.join(
                ['{} : {}'.format(k, v) for k, v in value.items()]))
      else:
        config_str += '\n>>    {} : {}'.format(key, value)
  return config_str

class NndctDebugger:

  def __init__(self):
    self.__DebugLv = 0

  def __host_info(self):
    return "<{}> ".format(self.__class__.__name__)

  def set_debug_lv(self, level):
    self.__DebugLv = level

  def debug(self, string, title='', level=None):
    nndct_debug_print(
        self.__host_info() + string, title=title, level=level or self.__DebugLv)

  def debug_details(self, obj, title, level=None):
    nndct_details_debug(
        obj,
        title=self.__host_info() + title,
        level=level or NNDCT_DEBUG_LVL.DETAILS)
