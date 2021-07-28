# Copyright 2021 Xilinx Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .dpuMcParserBase import *


class dpuv2InstParser(dpuMcParserBase):
    def __init__(self):
        name = "DPUCZDX8G_ISA0"
        super().__init__(name)
        self.opcode_table = [
            self.inst_desc("LOAD", 0b0000, 4),
            self.inst_desc("SAVE", 0b0100, 4),
            self.inst_desc("CONV", 0b1000, 5),
            self.inst_desc("CONVINIT", 0b1001, 4),
            self.inst_desc("DPTWISE", 0b1010, 5),
            self.inst_desc("DWINIT", 0b1011, 3),
            self.inst_desc("POOLINIT", 0b0110, 2),
            self.inst_desc("POOL", 0b1100, 5),
            self.inst_desc("ELEWINIT", 0b1101, 2),
            self.inst_desc("ELEW", 0b1110, 3),
            self.inst_desc("END", 0b0111, 1)
        ]

    class Load(LittleEndianStructure):
        _pack_ = 1
        _fields_ = [
            ('bank_addr', c_uint, 12),
            ('bank_id', c_uint, 6),
            ('hp_id', c_uint, 2),
            ('dpby', c_uint, 4),
            ('dpdon', c_uint, 4),
            ('opcode', c_uint, 4),

            ('jump_read', c_uint, 16),
            ('pad_idx', c_uint, 5),
            ('pad_end', c_uint, 5),
            ('pad_start', c_uint, 5),
            ('r0', c_uint, 1),

            ('channel', c_uint, 12),
            ('mode_avg', c_uint, 2),
            ('length', c_uint, 10),
            ('jump_write', c_uint, 8),

            ('ddr_addr', c_uint, 29),
            ('reg_id', c_uint, 3)
        ]

    class Save(LittleEndianStructure):
        _pack_ = 1
        _fields_ = [
            ('bank_addr', c_uint, 12),
            ('bank_id', c_uint, 6),
            ('hp_id', c_uint, 2),
            ('dpby', c_uint, 4),
            ('dpdon', c_uint, 4),
            ('opcode', c_uint, 4),

            ('jump_write', c_uint, 16),
            ('r0', c_uint, 16),

            ('channel', c_uint, 12),
            ('r1', c_uint, 2),
            ('length', c_uint, 10),
            ('jump_read', c_uint, 8),

            ('ddr_addr', c_uint, 29),
            ('reg_id', c_uint, 3)
        ]


register(dpuv2InstParser())
