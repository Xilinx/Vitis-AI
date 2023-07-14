# Copyright 2022-2023 Advanced Micro Devices Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

PREC = 3
TINY_FLG = "~0"


def to_Ki(_num):
    return _num / 1000.0


def to_Mi(_num):
    return _num / 1000.0 / 1000.0


def to_Gi(_num):
    return _num / 1000.0 / 1000.0 / 1000.0


def to_KB(_num):
    return _num / 1000.0


def to_MB(_num):
    return _num / 1000.0 / 1000.0


def to_GB(_num):
    return _num / 1000.0 / 1000.0 / 1000.0


"""
output_type: float/int/str
unit: Ki/Mi/Gi/KB/MB/GB

"""


class uConv:
    def __init__(self, _raw_num, _unit="", _prec=PREC):
        self.num = _raw_num
        self.o_num = self.num
        self.prec = _prec
        self.min = 10**(-_prec)

        _unit = _unit.lower()
        self.unit = _unit

        if _unit == "ki":
            self.o_num = float(self.num) / 10**3
        elif _unit == "mi":
            self.o_num = float(self.num) / 10**6
        elif _unit == "gi":
            self.o_num = float(self.num) / 10**9
        elif _unit == "kb":
            self.o_num = float(self.num) / 2**10
        elif _unit == "mb":
            self.o_num = float(self.num) / 2**20
        elif _unit == "gb":
            self.o_num = float(self.num) / 2**30
        elif _unit == "%":
            self.o_num = float(self.num) * 100
        else:
            pass

    def __str__(self):
        if self.o_num < self.min:
            return TINY_FLG
        else:
            fmt = "{:.%df}" % self.prec
            return fmt.format(self.o_num)

    def __float__(self):
        return self.o_num

    def __int__(self):
        return int(o_num)

    def __format__(self, code):
        if code == "" or code.endswith('s'):
            return self.__str__()

        fmt = "%%%s" % code
        return fmt % self.o_num
