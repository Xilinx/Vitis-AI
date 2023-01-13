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

import re
from vai_utf.python.target_factory import VAI_UTF as utf

class DPUTargetHelper(object):

  @staticmethod
  def get_basic_info(dpu_target):
    return f"name: {dpu_target.name}\ntype: {dpu_target.type}\nisa_version: {dpu_target.isa_version}"

  @staticmethod
  def get_full_info(dpu_target):
    return dpu_target._legacy_dpu_target_def
  
  @staticmethod
  def parse_range(num_range):
    new_range = []
    token = ","
    single_pattern = re.compile(r"""\s*(\d+)\s*""")
    range_pattern = re.compile(r"""\s*(\d+)\s*-\s*(\d+)\s*""")
    for num_item in num_range.split(token):
      if "-" in num_item:
        result = range_pattern.match(num_item)
        lower = int(result.group(1))
        upper = int(result.group(2))
        new_range.extend(list(range(lower, upper + 1)))
      else:
        result = single_pattern.match(num_item)
        num = int(result.group(1))
        new_range.append(num)
    return new_range

  @staticmethod
  def has_attr(message, member):
    assert hasattr(message, "ByteSize")
    return hasattr(message, member) and getattr(message, member).ByteSize() > 0

  @staticmethod
  def get_name(dpu_target):
    return dpu_target.get_name()

  @staticmethod
  def get_type(dpu_target):
    return dpu_target.get_type()
    
  @staticmethod
  def get_conv_engine(dpu_target):
    return dpu_target.get_conv_engine()
  
  @staticmethod
  def get_alu_engine(dpu_target):
    return dpu_target.get_alu_engine()

  @staticmethod
  def get_pool_engine(dpu_target):
    return dpu_target.get_pool_engine()

  @staticmethod
  def get_eltwise_engine(dpu_target):
    return dpu_target.get_eltwise_engine()

  @staticmethod
  def has_alu_engine(dpu_target):
    return hasattr(dpu_target, "alu_engine") and dpu_target.get_alu_engine().ByteSize() > 0

  @staticmethod
  def has_pool_engine(dpu_target):
    return hasattr(dpu_target, "pool_engine") and dpu_target.get_pool_engine().ByteSize() > 0
  
  @staticmethod
  def has_dwconv_engine(dpu_target):
    return hasattr(dpu_target, "dwconv_engine") and dpu_target.get_dwconv_engine().ByteSize() > 0
  
  @staticmethod
  def has_eltwise_engine(dpu_target):
    return hasattr(dpu_target, "eltwise_engine") and dpu_target.get_eltwise_engine().ByteSize() > 0

  @staticmethod
  def get_bank_group(dpu_target):
    return dpu_target.get_bank_group()
  
  @staticmethod
  def get_load_engine(dpu_target):
    return dpu_target.get_load_engine()
  
  @staticmethod
  def get_dwconv_engine(dpu_target):
    return dpu_target.get_dwconv_engine()
  