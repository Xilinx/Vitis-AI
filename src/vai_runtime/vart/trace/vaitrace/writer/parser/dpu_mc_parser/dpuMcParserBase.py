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
import logging
import copy
from ctypes import *
import logging
import binascii


class dpuMcParserBase:
    def __init__(self, _name="unknow"):
        self.name = _name
        self.data = {"load_img_size": 0, "load_para_size": 0,
                     "save_size": 0, "workload": 0}
        self.debug = False

    def compatible(self, dpu_name):
        if dpu_name.startswith(self.name):
            logging.debug("Compatible [%s]: True" % self.name)
            return True

        return False

    def get_inst(self, _mc, pos, dpu_cls):
        head = self.inst_head()
        memmove(addressof(head), _mc[pos:], sizeof(head))

        opcode = head.opcode
        for op in dpu_cls.opcode_table:
            if op.opcode == opcode:
                return (op.name, op.length_byte)

        if self.debug:
            raise ValueError("Unrecognized instruction")

        logging.error("Unrecognized instruction")
        return ("UNKNOW", 0)

    def process(self, mc, _debug=False):

        self.debug = _debug

        pos = 0
        load_img_size = 0
        load_para_size = 0
        save_size = 0

        while (pos < len(mc)):
            inst_name, inst_len = self.get_inst(mc, pos, self)

            if self.debug:
                print("pos: %d/%d, inst: %s          " %
                      (pos, len(mc), inst_name), end="\r")
            if inst_name == "LOAD":
                l = self.Load()
                inst = mc[pos:]
                memmove(addressof(l), inst, inst_len)

                length_ = l.length + 1
                chan_ = l.channel + 1
                try:
                    block_num_ = l.block_num + 1
                except:
                    block_num_ = 1

                #print("LOAD,%d,%d,%d" % (length_, chan_,block_num_))
                load_img_size += (length_ * chan_ * block_num_)

            elif inst_name == "SAVE":
                s = self.Save()
                inst = mc[pos:]
                memmove(addressof(s), inst, inst_len)

                length_ = s.length + 1
                chan_ = s.channel + 1
                save_size += (length_ * chan_)
                #print("SAVE %d %d" % (length_, chan_))
            elif inst_name == "UNKNOW":
                load_img_size = 0
                load_para_size = 0
                save_size = 0
                break
            else:
                pass

            pos += inst_len

        self.data["load_img_size"] = load_img_size
        self.data["load_para_size"] = load_para_size
        self.data["save_size"] = save_size

    """
    by default, return "load_img_size, load_para_size, save_size, workload",
    this method can be overrde
    """

    def get_data(self, items: list) -> list:
        ret = []
        ret.append(self.data.get("load_img_size"))
        ret.append(self.data.get("load_para_size"))
        ret.append(self.data.get("save_size"))
        ret.append(self.data.get("workload"))

        return ret

    class inst_head(LittleEndianStructure):
        _pack_ = 1
        _fields_ = [
            ('resverd', c_uint, 20),
            ('dpby', c_uint, 4),
            ('dpdon', c_uint, 4),
            ('opcode', c_uint, 4)
        ]

    class inst_desc():
        def __init__(self, _name, _opcode, _length_w):
            self.name = _name
            self.opcode = _opcode
            self.length_w = _length_w
            self.length_byte = _length_w * 4


__dpu_mc_parser = []


"""
by default, reture [load_img_size, load_para_size, save_size, workload]
"""


def process_mc(dpu_name: str, mc_data_str: str, _debug=True, ret_items=["load_img_size", "load_para_size", "save_size", "workload"]) -> list:
    # init return data with 0
    ret = []
    for i in range(len(ret_items)):
        ret.append(0)

    # 1. find dpu mc parser
    matched = False
    for parser in __dpu_mc_parser:
        if parser.compatible(dpu_name) == False:
            continue
        matched = True
        logging.debug("Processing DPU mc [%s]..." % parser.name)

        # 2. process_mc
        mc = binascii.a2b_hex(mc_data_str)
        parser.process(mc, _debug)

        # 3. get_data
        ret = parser.get_data(ret_items)

    if matched == False:
        logging.warning(
            "Can't find dpu mc parser for %s, return all 0" % dpu_name)

    return ret


def register(mcParserInstance):
    __dpu_mc_parser.append(mcParserInstance)
