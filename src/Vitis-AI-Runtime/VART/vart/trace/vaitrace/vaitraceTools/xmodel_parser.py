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

from functools import reduce
import sys
import os
import logging
import copy
from ctypes import *
import xir

SUBGRAPH_DB = {}
DPU_NAME = ""


class Load(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('opcode', c_uint, 4),
        ('dpdon', c_uint, 4),
        ('dpby', c_uint, 4),
        ('bank_id', c_uint, 6),
        ('bank_addr', c_uint, 14),
        ('r0', c_uint, 1),
        ('pad_start', c_uint, 5),
        ('pad_end', c_uint, 5),
        ('pad_idx', c_uint, 5),
        ('jump_read', c_uint, 16),
        ('jump_write', c_uint, 8),
        ('length', c_uint, 10),
        ('mode_avg', c_uint, 2),
        ('channel', c_uint, 12),
        ('reg_id', c_uint, 3),
        ('ddr_addr', c_uint, 29),
        ('r1', c_uint, 8),
        ('block_num', c_uint, 10),
        ('jump_write_endl', c_uint, 14)
    ]


class Save(BigEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('opcode', c_uint, 4),
        ('dpdon', c_uint, 4),
        ('dpby', c_uint, 4),
        ('bank_id', c_uint, 6),
        ('bank_addr', c_uint, 14),
        ('r0', c_uint, 16),
        ('jump_write', c_uint, 16),
        ('jump_read', c_uint, 8),
        ('length', c_uint, 10),
        ('r1', c_uint, 2),
        ('channel', c_uint, 12),
        ('reg_id', c_uint, 3),
        ('ddr_addr', c_uint, 29)
    ]


opcode_table = [
    {
        'name': "LOAD",
        'opcode': 0b0000,
        'length_w': 5,
        'layout': ""
    }, {
        'name': "SAVE",
        'opcode': 0b0100,
        'length_w': 4,
        'layout': ""
    }, {
        'name': "CONV",
        'opcode': 0b1000,
        'length_w': 5
    }, {

        'name': "CONVINIT",
        'opcode': 0b1001,
        'length_w': 6
    }, {

        'name': "DPTWISE",
        'opcode': 0b1010,
        'length_w': 5
    }, {
        'name': "DWINIT",
        'opcode': 0b1011,
        'length_w': 4
    }, {

        'name': "POOLINIT",
        'opcode': 0b0110,
        'length_w': 2
    }, {

        'name': "POOL",
        'opcode': 0b1100,
        'length_w': 5
    }, {

        'name': "ELEWINIT",
        'opcode': 0b1101,
        'length_w': 2
    }, {

        'name': "ELEW",
        'opcode': 0b1110,
        'length_w': 3
    }, {

        'name': "END",
        'opcode': 0b0111,
        'length_w': 1
    }
]


def process_mc(mc):
    global DPU_NAME
    if not DPU_NAME.startswith("DPUCVDX8G_ISA0"):
        return 0, 0

    pos = 0

    def get_next_op(_mc):
        opcode = _mc[pos+3] >> 4
        load_size = 0
        save_size = 0

        for op in opcode_table:
            if op['opcode'] == opcode:
                mc_name = op['name']
                if mc_name == "LOAD":
                    l = Load()
                    byte_length = op['length_w']
                    for i in range(0, byte_length):
                        for j in range(0, 4):
                            dest_start = addressof(l)
                            dest_off = i * 4 + j
                            src_off = i * 4 + 3 - j
                            src_buf = _mc[pos + src_off]
                            memset(dest_start+dest_off, src_buf, 1)

                    length_ = l.length + 1
                    chan_ = l.channel + 1
                    block_num_ = l.block_num + 1
                    #print("LOAD,%d,%d,%d" % (length_, chan_, block_num_))
                    load_size += (length_ * chan_ * block_num_)

                if mc_name == "SAVE":
                    s = Save()
                    byte_length = op['length_w']
                    for i in range(0, byte_length):
                        for j in range(0, 4):
                            dest_start = addressof(s)
                            dest_off = i * 4 + j
                            src_off = i * 4 + 3 - j
                            src_buf = _mc[pos + src_off]
                            memset(addressof(s)+dest_off, src_buf, 1)

                    length_ = s.length + 1
                    chan_ = s.channel + 1
                    save_size += (length_ * chan_)
                    #print("SAVE %d %d" % (length_, chan_))

                return op['length_w'] * 4, load_size, save_size
        assert(False)

    load_size = 0
    save_size = 0
    while (pos < len(mc)):
        offsit, load_s, save_s = get_next_op(mc)
        pos += offsit
        load_size += load_s
        save_size += save_s
    return load_size, save_size


def to_unit(value, unit="m", precision=3):
    if unit.lower().startswith("k"):
        value = value / 1024
    elif unit.lower().startswith("m"):
        value = value / 1024 / 1024
    elif unit.lower().startswith("g"):
        value = value / 1024 / 1024 / 1024
    else:
        raise("Unit format error")

    return "%%.%df" % precision % value


def find_in_inc(inc, key):
    value = 0
    incs = inc.split()
    for i in incs:
        if i == key:
            idx = incs.index(i)
    value = incs[idx+1]
    return value


def inc_get_io_tensor_size(inc):
    ch = 1
    length = 1
    block_num = 1
    if inc.startswith("LOAD"):
        ch = int(find_in_inc(inc, 'channel'))
        length = int(find_in_inc(inc, "length"))
        block_num = int(find_in_inc(inc, 'block_num'))
    elif inc.startswith("SAVE"):
        ch = int(find_in_inc(inc, 'channel'))
        length = int(find_in_inc(inc, "length"))
    else:
        pass

    return ch * length * block_num


def subg_to_id(sub_g, key="workload"):
    if key == "workload":
        workload = 0
        name = sub_g.get_name()
        if (sub_g.has_attr("workload")):
            workload = sub_g.get_attr("workload")
        return "%s|%d" % (name, workload)
    if key == "depth":
        name = sub_g.get_name()
        depth = sub_g.depth
        return "%s|%d" % (name, depth)


def idx_to_name(idx):
    return idx.rsplit('|', 2)[0]


def get_subg_info(sub_g):
    global DPU_NAME
    global SUBGRAPH_DB

    name = sub_g.get_name().strip()
    idx_w = subg_to_id(sub_g, "workload")

    workload = 0
    op_num = 0
    load_io_size = 0
    save_io_size = 0
    i_tensor_shape = []
    o_tensor_shape = []
    device = "unknow"

    i_tensors = sub_g.get_input_tensors()
    for it in i_tensors:
        i_tensor_shape.append(it.dims)
    o_tensors = sub_g.get_output_tensors()
    for ot in o_tensors:
        o_tensor_shape.append(ot.dims)

    if (sub_g.has_attr("dpu_name")):
        DPU_NAME = sub_g.get_attr("dpu_name")

    op_num = sub_g.get_op_num()
    if (sub_g.has_attr("workload")):
        workload = sub_g.get_attr("workload")

    if (sub_g.has_attr("device")):
        device = sub_g.get_attr("device")

    if (sub_g.has_attr("ac_code")):
        ac = sub_g.get_attr("ac_code")
        load_incs = [i for i in ac if i.startswith('LOAD')]
        for i in load_incs:
            load_io_size += inc_get_io_tensor_size(i)
        save_incs = [i for i in ac if i.startswith('SAVE')]
        for i in save_incs:
            save_io_size += inc_get_io_tensor_size(i)

    elif (sub_g.has_attr("mc_code")):
        mc = sub_g.get_attr("mc_code")
        load_io_size, save_io_size = process_mc(mc)

    subg_info = {idx_w:
                 {"name": name,
                     "device": device,
                     "workload": workload,
                     "op_num": op_num,
                     "load_io_size": load_io_size,
                     "save_io_size": save_io_size,
                     "i_tensor_shape": i_tensor_shape,
                     "o_tensor_shape": o_tensor_shape}}
    if idx_w not in SUBGRAPH_DB.keys():
        SUBGRAPH_DB.update(subg_info)
    else:
        logging.warning("SubGraph index duplication: [%s]" % idx_w)

    for sub in sub_g.get_children():
        get_subg_info(sub)


def xmodel_get_info(xmodel: []):
    SUBGRAPH_DB.clear()

    for m in xmodel:
        if not os.path.exists(m):
            raise("Error XModel Path: %s" % m)

        graph = xir.Graph.deserialize(m)
        root = graph.get_root_subgraph()
        get_subg_info(root)

    return SUBGRAPH_DB
