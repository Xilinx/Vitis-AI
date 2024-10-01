"""
 Copyright 2019 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from enum import Enum


class LayoutType(Enum):
    INSENSITIVE = 1
    TOLERANT = 2
    DEPENDENT = 3
    RECONSTRUCTED = 4


class Layout(Enum):
    NCHW = 0
    NHWC = 1


class RoundMode(Enum):
    FLOOR = 0
    CEIL = 1
    STD_ROUND = 2
    PYTHON3_ROUND = 3
    ROUND_DOWN = 0
    ROUND_UP = 1
    ROUND_AWAY_FROM_ZERO = 2
    ROUND_HALF_TO_EVEN = 3


class PadMode(Enum):
    EXPLICIT = 0
    SAME = 1
    VALID = 2


class TargetType(Enum):
    OPENIR = 0
    XIR = 1
