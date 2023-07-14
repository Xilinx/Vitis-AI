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

from functools import reduce
import sys
import os
import logging
import copy
from ctypes import *
import json
import logging
import binascii

from . import dpu_mc_parser


SUBGRAPH_DB = {}
DPU_NAME = ""


class subgraph_info:
    def __init__(self):
        pass


class xmodel_info:
    def __init__(self, name):
        self.name = name
        self.subgraphs = []


def subg_to_id(sub_g, key="workload"):
    if key == "workload":
        workload = 0
        name = sub_g.get("subgraph_name")
        workload = sub_g.get("workload")
        return "%s|%d" % (name, workload)
    if key == "depth":
        name = sub_g.get("subgraph_name")
        depth = sub_g.get("depth")
        return "%s|%d" % (name, depth)


def idx_to_name(idx):
    return idx.rsplit('|', 2)[0]


def get_subg_info(sub_g: dict, rank_id):
    """ Example subg data
    {
    "subgraph_name":"subgraph_pool5","id":"0-3","dpu_name":"","device":"","workload":100352,"op_num":1,"i_tensors_shape":[[1,7,7,2048]],"o_tensors_shape":[[1,1,1,2048]],
    "mc_code_sstr.str()":
        "a0021001ff070000ff87017f004c02
         20a0121001ff070000ff87017f0084
         ...
        }
    """
    global DPU_NAME
    global SUBGRAPH_DB

    """Only update DPU IP name for once"""
    if (DPU_NAME == ""):
        DPU_NAME = sub_g.get("dpu_name", "")

    idx_w = subg_to_id(sub_g, "workload")

    name = sub_g.get("subgraph_name").strip()
    workload = sub_g.get("workload", 0)
    depth = sub_g.get("depth", 0)
    op_num = sub_g.get("op_num", 0)
    depth = sub_g.get("depth", 0)
    device = sub_g.get("device", "unknow")

    mc = sub_g.get("mc_code_sstr.str()", "")
    mc_size = len(mc) / 2 # div 2: sizeof string -> sizeof bin

    if (mc_size < 100_000_000):
        if (mc_size > 50_000_000):
            logging.warning(
                "vaitrace is analyzing .xmodel, this model is large, please wait for a while ")
        mc_info = dpu_mc_parser.process_mc(DPU_NAME, mc)
    else:
        logging.error(
            "vaitrace is analyzing .xmodel, this model is too large, vaitrace can not handle it now ")
        mc_info = [0, 0, 0, 0]

    load_io_img_size = mc_info[0]
    load_io_para_size = mc_info[1]
    save_io_size = mc_info[2]

    i_tensor_shape = sub_g.get("i_tensors_shape", [])
    o_tensor_shape = sub_g.get("o_tensors_shape", [])

    subg_info = {idx_w: {
                  "name": name,
                  "device": device,
                  "workload": workload,
                  "op_num": op_num,
                  "depth": depth,
                  "mc_size": mc_size,
                  "load_io_img_size": load_io_img_size,
                  "load_io_para_size": load_io_para_size,
                  "save_io_size": save_io_size,
                  "i_tensor_shape": i_tensor_shape,
                  "o_tensor_shape": o_tensor_shape,
                  "rank_id": rank_id}}

    if idx_w not in SUBGRAPH_DB.keys():
        SUBGRAPH_DB.update(subg_info)
    else:
        logging.info("SubGraph index duplication: [%s]" % idx_w)


"""input: {'classname': 'subgraph_info', 'info_file': '/tmp/vaitrace_subgraph_info_15230', 'section': 'INFO'}"""


def xmodel_get_info(raw_trace: []):
    SUBGRAPH_DB.clear()
    raw_x_info = []

    if len(raw_trace) == 0:
        logging.error("Xmodel info raw data empty")
        return {}
    try:
        for item in raw_trace:
            info_file = item.get('info_file')
            logging.debug("Open xmodel information file: %s" % info_file)
            subgs = open(info_file, "r+t").readlines()
            logging.debug("Subgraphs to process: %d" % len(subgs))
            for subg in subgs:
                raw_x_info.append(subg)
    except:
        logging.error("xmodel parser error")
        return {}

    logging.info("Processing xmodel information")
    rank_id = 0
    for subg_str in raw_x_info:
        try:
            subg = json.loads(subg_str)
        except:
            logging.warning("Subgraph information format error")

        get_subg_info(subg, rank_id)
        rank_id += 1

    return SUBGRAPH_DB
