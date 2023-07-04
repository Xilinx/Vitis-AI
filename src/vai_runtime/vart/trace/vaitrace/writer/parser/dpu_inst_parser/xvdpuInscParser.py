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

import sys
import os
from ctypes import *


class InstDumpHeader(LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('bank_addr', c_uint, 13),
        ('output_channel_num', c_uint, 7),
        ('dpby', c_uint, 4),
        ('dpdon', c_uint, 4),
        ('opcode', c_uint, 4),


        ('jump_read', c_uint, 16),
        ('bank_id', c_uint, 8),
        ('ddr_mode', c_uint, 1),
        ('pad_idx', c_uint, 5),
        ('r0', c_uint, 2),

        ('channel', c_uint, 12),
        ('mode_avg', c_uint, 2),
        ('length', c_uint, 10),
        ('jump_write', c_uint, 8),

        ('ddr_addr', c_uint, 29),
        ('reg_id', c_uint, 3),

        ('jump_read_endl', c_uint, 21),
        ('pad_end', c_uint, 6),
        ('pad_start', c_uint, 5)
    ]


class InstDumpItem(LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('bank_addr', c_uint, 13),
        ('r0', c_uint, 7),
        ('dpby', c_uint, 4),
        ('dpdon', c_uint, 4),
        ('opcode', c_uint, 4),

        ('jump_write', c_uint, 16),
        ('bank_id', c_uint, 8),
        ('r1', c_uint, 8),

        ('channel', c_uint, 12),
        ('r2', c_uint, 2),
        ('length', c_uint, 10),
        ('jump_read', c_uint, 8),

        ('ddr_addr', c_uint, 29),
        ('reg_id', c_uint, 3)
    ]


data = open("./profiler__batch_0_instr_all_start_0.bin", "rb").read()
