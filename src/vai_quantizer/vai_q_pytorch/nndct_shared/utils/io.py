

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

import os
import shutil
import json
import sys
import numpy as np

from .log import log_or_print
from nndct_shared.base import NNDCT_KEYS, GLOBAL_MAP



def to_des_dict(dicts, as_des=True, extra_types={}):
  assert isinstance(dicts,list) and all(isinstance(d,dict) for d in dicts),\
      "dicts should be list of dictionaries, please check!"
  des_dict = {}
  for d in dicts:
    for k, v in d.items():
      if isinstance(v, np.ndarray):
        v = v.tolist()
      elif isinstance(v, str) and not k in ['dtype'] and as_des:
        v = "'{}'".format(v)
      elif k == 'shape' and v == []:
        continue
      elif isinstance(v, dict):
        v = to_des_dict([v], as_des, extra_types)
      for typek, func in extra_types.items():
        if isinstance(v, typek):
          v = func(v)
      des_dict[k] = v
  return des_dict

def dict_to_str(kwargs, connect='=', as_des=False):
  if as_des:
    kwargs = to_des_dict([kwargs])
  return ','.join(["{}{}{}".format(k, connect, v) for k, v in kwargs.items()])

def check_diff(matA, matB, nameA, nameB, error, with_msg=True):
  is_pass = True
  mat = matA - matB
  mat = mat / np.sqrt(matA**2 + error)
  title = "{:25} VS {:25} : ".format(nameA, nameB)
  res = {}
  string = ''
  res['max'] = mat.max()
  res['min'] = mat.min()
  for item in res:
    string += item + ':' + str(res[item]) + ' '
    if abs(res[item]) > error:
      msg = "{}({}) out of tolerance({})!".format(item, res[item], error)
      is_pass = False
      break
  if with_msg:
    if is_pass:
      msg = string + '(tolerance {})'.format(error)
      print("{}{}{:60}{}".format('**', '[PASS] ' + title, msg, '**'))
    else:
      print("{}{}{:60}{}".format('**', '[FAIL] ' + title, msg, '**'))
  return is_pass

def to_jsonstr(obj, pre_space=2):

  def _json_lst_str(lst):
    string = ""
    for idx in range(len(lst)):
      if isinstance(lst[idx], str):
        string += '"{}",'.format(lst[idx])
      elif isinstance(lst[idx], list):
        #string += _json_lst_str(lst[idx])
        string += '[{}],'.format(_json_lst_str(lst[idx]))
      elif lst[idx] is None:
        string += 'null,'
      else:
        string += str(lst[idx]) + ','
    return string[:-1]

  assert isinstance(obj, dict)
  string = ""
  for k, v in obj.items():
    string += '{}"{}":'.format(pre_space * ' ', k)
    if isinstance(v, list):
      string += '[{}],\n'.format(_json_lst_str(v))
    elif isinstance(v, str):
      string += '"{}",\n'.format(v)
    elif isinstance(v, dict):
      string += '\n{},\n'.format(to_jsonstr(v, pre_space=pre_space + 2))
    elif isinstance(v, bool):
      string += '{},\n'.format('true' if v else 'false')
    else:
      string += '{},\n'.format(v)
  return '{}{{\n{}\n{}}}'.format((pre_space - 2) * ' ', string[:-2],
                                 (pre_space - 2) * ' ')

def load_json_obj(file_or_obj):
  if isinstance(file_or_obj, str):
    with open(file_or_obj, 'r') as f:
      obj = json.load(f)
  elif isinstance(file_or_obj, dict):
    obj = file_or_obj
  else:
    return None
  return obj

def dpu_format_print(mat):
  flatten_mat = mat.reshape(mat.size)
  cnt = 0
  while cnt < len(flatten_mat):
    print(("{:0>2x}" * 16).format(*tuple(flatten_mat[cnt:cnt + 16])))
    cnt += 16

def copy_folder_files(new_dir, old_dir):
  for file_name in os.listdir(old_dir):
    full_file_name = os.path.join(old_dir, file_name)
    if (os.path.isfile(full_file_name)):
      shutil.copy(full_file_name, new_dir)

def force_create_dir(dir_name, copy_from_dir=None):
  if os.path.exists(dir_name):
    shutil.rmtree(dir_name)
  os.makedirs(dir_name)
    
  if copy_from_dir:
    copy_folder_files(dir_name, copy_from_dir)

def create_work_dir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    
 

def print_center_edge(string, to_str=False, blank_line=0, width=120):
  center_str = "{0}>>{1:40}<<{0}".format("=" * 30,
                                         string.center(40)).center(width)
  center_str += '\n' * blank_line
  if to_str:
    return center_str
  else:
    print(center_str)

def basic_info(mat, name=None, logger=None, to_str=False):
  if isinstance(mat, np.ndarray):
    info_str = "<Array>{}[{}]: max:{}, min:{}, sum:{}".format(
        '' if not name else name, mat.shape, mat.max(), mat.min(), mat.sum())
  else:
    info_str = "<Non_Array>{}:{}".format('' if not name else name, mat)
  if to_str:
    return info_str
  else:
    log_or_print(info_str, logger=logger)

def print_csv_format(mat):
  assert mat.ndim == 2
  for row in range(mat.shape[0]):
    for col in range(mat.shape[1]):
      print(str(mat[row, col]) + ',', end='')
    print('')

def print_mat(mat, name="tmp mat", col=20, t=0, channel=sys.stdout):
  max_val = mat.max()
  min_val = mat.min()
  sum_val = mat.sum()
  if isinstance(max_val, float) or isinstance(max_val, int):
    pass
  else:
    max_val = max_val[0]
    min_val = min_val[0]
    sum_val = sum_val[0]
  if len(mat.shape) == 2:
    channel.write("showing " + str(t) + " th " + name + str(mat.shape) + "\n")
    channel.write("sum:" + str(sum_val) + "; max:" + str(max_val) + " min:" +
                  str(min_val) + "\n")
    for r in range(mat.shape[0]):
      for c in range(col):
        channel.write(str(mat[r, c]) + " ")
      channel.write("\n")
  elif len(mat.shape) == 1:
    channel.write("showing " + str(t) + " th " + name + "\n")
    channel.write("sum:" + str(sum_val) + "; max:" + str(max_val) + " min:" +
                  str(min_val) + "\n")
    for r in range(col):
      channel.write(str(mat[r]) + " ")
    channel.write("\n")
  else:
    raise Exception("wrong mode!")

def latest_file(folder, file_checker=None):
  lists = os.listdir(folder)
  lists.sort(key=lambda fn: os.path.getmtime(os.path.join(folder, fn)))
  for file_name in lists[::-1]:
    abs_file = os.path.join(folder, file_name)
    if not file_checker or file_checker(abs_file):
      return abs_file
