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


# This code is based on: https://github.com/nutonomy/second.pytorch.git
# 
# MIT License
# Copyright (c) 2018 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import contextlib
import enum
import math
import time

import numpy as np


def progress_str(val, *args, width=20, with_ptg=True):
    val = max(0., min(val, 1.))
    assert width > 1
    pos = round(width * val) - 1
    if with_ptg is True:
        log = '[{}%]'.format(max_point_str(val * 100.0, 4))
    log += '['
    for i in range(width):
        if i < pos:
            log += '='
        elif i == pos:
            log += '>'
        else:
            log += '.'
    log += ']'
    for arg in args:
        log += '[{}]'.format(arg)
    return log


def second_to_time_str(second, omit_hours_if_possible=True):
    second = int(second)
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    if omit_hours_if_possible:
        if h == 0:
            return '{:02d}:{:02d}'.format(m, s)
    return '{:02d}:{:02d}:{:02d}'.format(h, m, s)


def progress_bar_iter(task_list, width=20, with_ptg=True, step_time_average=50, name=None):
    total_step = len(task_list)
    step_times = []
    start_time = 0.0
    name = '' if name is None else f"[{name}]"
    for i, task in enumerate(task_list):
        t = time.time()
        yield task
        step_times.append(time.time() - t)
        start_time += step_times[-1]
        start_time_str = second_to_time_str(start_time)
        average_step_time = np.mean(step_times[-step_time_average:]) + 1e-6
        speed_str = "{:.2f}it/s".format(1 / average_step_time)
        remain_time = (total_step - i) * average_step_time
        remain_time_str = second_to_time_str(remain_time)
        time_str = start_time_str + '>' + remain_time_str
        prog_str = progress_str(
            (i + 1) / total_step,
            speed_str,
            time_str,
            width=width,
            with_ptg=with_ptg)
        print(name + prog_str + '   ', end='\r')
    print("")


list_bar = progress_bar_iter

def enumerate_bar(task_list, width=20, with_ptg=True, step_time_average=50, name=None):
    total_step = len(task_list)
    step_times = []
    start_time = 0.0
    name = '' if name is None else f"[{name}]"
    for i, task in enumerate(task_list):
        t = time.time()
        yield i, task
        step_times.append(time.time() - t)
        start_time += step_times[-1]
        start_time_str = second_to_time_str(start_time)
        average_step_time = np.mean(step_times[-step_time_average:]) + 1e-6
        speed_str = "{:.2f}it/s".format(1 / average_step_time)
        remain_time = (total_step - i) * average_step_time
        remain_time_str = second_to_time_str(remain_time)
        time_str = start_time_str + '>' + remain_time_str
        prog_str = progress_str(
            (i + 1) / total_step,
            speed_str,
            time_str,
            width=width,
            with_ptg=with_ptg)
        print(name + prog_str + '   ', end='\r')
    print("")


def max_point_str(val, max_point):
    positive = bool(val >= 0.0)
    val = np.abs(val)
    if val == 0:
        point = 1
    else:
        point = max(int(np.log10(val)), 0) + 1
    fmt = "{:." + str(max(max_point - point, 0)) + "f}"
    if positive is True:
        return fmt.format(val)
    else:
        return fmt.format(-val)


class Unit(enum.Enum):
    Iter = 'iter'
    Byte = 'byte'


def convert_size(size_bytes):
    # from https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return s, size_name[i]


class ProgressBar:
    def __init__(self,
                 width=20,
                 with_ptg=True,
                 step_time_average=50,
                 speed_unit=Unit.Iter):
        self._width = width
        self._with_ptg = with_ptg
        self._step_time_average = step_time_average
        self._step_times = []
        self._start_time = 0.0
        self._total_size = None
        self._speed_unit = speed_unit

    def start(self, total_size):
        self._start = True
        self._step_times = []
        self._finished_sizes = []
        self._time_elapsed = 0.0
        self._current_time = time.time()
        self._total_size = total_size
        self._progress = 0

    def print_bar(self, finished_size=1, pre_string=None, post_string=None):
        self._step_times.append(time.time() - self._current_time)
        self._finished_sizes.append(finished_size)
        self._time_elapsed += self._step_times[-1]
        start_time_str = second_to_time_str(self._time_elapsed)
        time_per_size = np.array(self._step_times[-self._step_time_average:])
        time_per_size /= np.array(
            self._finished_sizes[-self._step_time_average:])
        average_step_time = np.mean(time_per_size) + 1e-6
        if self._speed_unit == Unit.Iter:
            speed_str = "{:.2f}it/s".format(1 / average_step_time)
        elif self._speed_unit == Unit.Byte:
            size, size_unit = convert_size(1 / average_step_time)
            speed_str = "{:.2f}{}/s".format(size, size_unit)
        else:
            raise ValueError("unknown speed unit")
        remain_time = (self._total_size - self._progress) * average_step_time
        remain_time_str = second_to_time_str(remain_time)
        time_str = start_time_str + '>' + remain_time_str
        prog_str = progress_str(
            (self._progress + 1) / self._total_size,
            speed_str,
            time_str,
            width=self._width,
            with_ptg=self._with_ptg)
        self._progress += finished_size
        if pre_string is not None:
            prog_str = pre_string + prog_str
        if post_string is not None:
            prog_str += post_string
        if self._progress >= self._total_size:
            print(prog_str + '   ')
        else:
            print(prog_str + '   ', end='\r')
        self._current_time = time.time()
