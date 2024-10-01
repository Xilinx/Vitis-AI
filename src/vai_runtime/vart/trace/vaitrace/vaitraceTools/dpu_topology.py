#!/usr/bin/python3

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
import ctypes
import logging
import json
from ascii_table import *

"""   MEMORY TOPOLOGY SECTION   """


class mem_u1 (ctypes.Union):
    _fields_ = [
        ("m_size", ctypes.c_int64),
        ("route_id", ctypes.c_int64)
    ]


class mem_u2 (ctypes.Union):
    _fields_ = [
        ("m_base_address", ctypes.c_int64),
        ("flow_id", ctypes.c_int64)
    ]


class mem_data (ctypes.Structure):
    _anonymous_ = ("mem_u1", "mem_u2")
    _fields_ = [
        ("m_type", ctypes.c_uint8),
        ("m_used", ctypes.c_uint8),
        ("mem_u1", mem_u1),
        ("mem_u2", mem_u2),
        ("m_tag", ctypes.c_char * 16)
    ]


class mem_topology (ctypes.Structure):
    _fields_ = [
        ("m_count", ctypes.c_int32),
        ("m_mem_data", mem_data*1000)
    ]


"""   IP_LAYOUT SECTION   """


class IP_CONTROL:
    AP_CTRL_HS = 0
    AP_CTRL_CHAIN = 1
    AP_CTRL_NONE = 2
    AP_CTRL_ME = 3


class indices (ctypes.Structure):
    _fields_ = [
        ("m_index", ctypes.c_uint16),
        ("m_pc_index", ctypes.c_uint8),
        ("unused", ctypes.c_uint8)
    ]


class ip_u1 (ctypes.Union):
    _fields_ = [
        ("m_base_address", ctypes.c_int64),
        ("indices", indices)
    ]


class ip_data (ctypes.Structure):
    _fields_ = [
        ("m_type", ctypes.c_uint32),
        ("properties", ctypes.c_uint32),
        ("ip_u1", ip_u1),
        ("m_name", ctypes.c_uint8 * 64)
    ]


class ip_layout (ctypes.Structure):
    _fields_ = [
        ("m_count", ctypes.c_int32),
        ("m_ip_data", ip_data*16)
    ]


"""   CONNECTIVITY SECTION   """


class connection(ctypes.Structure):
    _fields_ = [
        ("arg_index", ctypes.c_int32),
        ("m_ip_layout_index", ctypes.c_int32),
        ("mem_data_index", ctypes.c_int32)
    ]


class connectivity(ctypes.Structure):
    _fields_ = [
        ("m_count", ctypes.c_int32),
        ("m_connection", connection*1000)
    ]


cu_args_name = {
    0:  {"name": "dpu_doneclr", "port": "S_AXI_CONTROL"},
    1:  {"name": "dpu_prof_en", "port": "S_AXI_CONTROL"},
    2:  {"name": "dpu_cmd", "port": "S_AXI_CONTROL"},
    3:  {"name": "dpu_instr_addr", "port": "M_AXI_GP0"},
    4:  {"name": "dpu_prof_addr", "port": "M_AXI_GP0"},
    5:  {"name": "dpu_base0_addr", "port": "M_AXI_HP0"},
    6:  {"name": "dpu_base1_addr", "port": "M_AXI_HP0"},
    7:  {"name": "dpu_base2_addr", "port": "M_AXI_HP0"},
    8:  {"name": "dpu_base3_addr", "port": "M_AXI_HP0"},
    9:  {"name": "dpu_base4_addr", "port": "M_AXI_HP2"},
    10: {"name": "dpu_base5_addr", "port": "M_AXI_HP2"},
    11: {"name": "dpu_base6_addr", "port": "M_AXI_HP2"},
    12: {"name": "dpu_base7_addr", "port": "M_AXI_HP2"}
}


def getIPsZynq(path):
    cus = {}
    ip_layout_data = open(path, 'rb').read()
    ips = ip_layout()

    ctypes.memmove(ctypes.pointer(ips), ip_layout_data, len(ip_layout_data))

    for i in range(0, ips.m_count):
        ip = ips.m_ip_data[i]
        cu_name = str()

        for c in ip.m_name:
            cu_name += chr(c)
        cu_type, cu_name = cu_name.strip('\x00').split(':')

        cu_name = cu_name.replace('_xrt_top', '')
        cu_paddr = hex(ip.ip_u1.m_base_address)
        cu_idx = ip.ip_u1.indices.m_index
        cu_pc_idx = ip.ip_u1.indices.m_pc_index

        #cus[cu_name] = (cu_pc_idx,cu_paddr)
        #cus[cu_pc_idx] = (cu_name,cu_paddr)
        cus[i] = (cu_name, cu_paddr)

    return {'cus': cus}


def getConnectivity(path):
    conn_data = open(path, 'rb').read()
    conn = connectivity()

    ret = {}

    ctypes.memmove(ctypes.pointer(conn), conn_data, len(conn_data))
    for i in range(0, conn.m_count):
        con = conn.m_connection[i]
        ret.setdefault(con.m_ip_layout_index, []).append(
            {'arg_index': con.arg_index, 'mem_data_index': con.mem_data_index})
        #print ("m_ip_layout_index %d, arg_index %d, mem_data_index %d" % (con.m_ip_layout_index, con.arg_index, con.mem_data_index))

    return {'connectivity': ret}


def getMemTopo(path):
    mem_topo_data = open(path, 'rb').read()
    mem_topo = mem_topology()

    ret = {}

    ctypes.memmove(ctypes.pointer(mem_topo), mem_topo_data, len(mem_topo_data))

    print(mem_topo.m_count)
    for i in range(mem_topo.m_count):
        print(
            f"{i}: {mem_topo.m_mem_data[i].m_tag}, {mem_topo.m_mem_data[i].m_type}, {mem_topo.m_mem_data[i].mem_u2.flow_id}")
    #xbutil_cmd = "xbutil"
    # for alt in xbutil_path_alt:
    #    if os.path.exists(alt):
    #        xbutil_cmd = alt

    #d = os.popen('%s --legacy dump' % xbutil_cmd).read()
    # if len(d) == 0:
    #    return {}

    #ret = json.loads(d)['board']["memory"]
    # return ret


def checkPath():
    alt_dirs = [
        "/sys/devices/platform/amba/amba:zyxclmm_drm/",
        "/sys/devices/platform/axi/axi:zyxclmm_drm/",
        "/sys/devices/platform/amba_pl@0/amba_pl@0:zyxclmm_drm/"
    ]

    for d in alt_dirs:
        if os.path.isdir(d):
            return d

    logging.error("Cannot find path amba:zyxclmm_drm")
    exit(-1)


path = checkPath()
ip_connect = getConnectivity(os.path.join(path, "connectivity"))
ip_layout = getIPsZynq(os.path.join(path, "ip_layout"))
mem_topo = getMemTopo(os.path.join(path, "mem_topology"))

print(ip_connect)
print(ip_layout)
print(mem_topo)
#
exit(-1)

DPU_TOPO = {}
for dpu_idx in ip_layout['cus'].keys():
    DPU_TOPO.setdefault(dpu_idx, {})['name'] = ip_layout['cus'][dpu_idx][0]
    DPU_TOPO.setdefault(dpu_idx, {})['address'] = ip_layout['cus'][dpu_idx][1]
    DPU_TOPO.setdefault(dpu_idx, {})[
        'ports'] = ip_connect['connectivity'][dpu_idx]
    for port in DPU_TOPO[dpu_idx]['ports']:
        port_idx = port['mem_data_index']
        port_idx_tag = mem_topo['mem'][str(port_idx)]['tag']

        arg_idx = int(port['arg_index'])
        arg_name = cu_args_name[arg_idx]['name']
        port['tag'] = port_idx_tag
        port['arg_name'] = arg_name


def print_dpu_info(dpu_info):

    title = ['CU Index', 'CU Name', 'CU Address',
             'Instruct Port', 'Data Ports']
    datas = []
    datas.append(title)

    for dpu_idx in dpu_info.keys():
        content = dpu_info[dpu_idx]

        cu_id = dpu_idx
        cu_name = content.get("name", "")
        cu_addr = content.get("address", "0x0")
        cu_ports = content.get("ports", [])

        inst_port = [p for p in cu_ports if p["arg_name"] == "dpu_instr_addr"]
        inst_port_tags = str(list(set([p["tag"] for p in inst_port]))).replace(
            "'", "").strip('[').strip(']')

        data_port = [
            p for p in cu_ports if p["arg_name"].find("dpu_base") >= 0]
        data_port_tags = str(list(set([p["tag"] for p in data_port]))).replace(
            "'", "").strip('[').strip(']')

        datas.append([dpu_idx, cu_name, cu_addr,
                      inst_port_tags, data_port_tags])

    print_ascii_table(datas)


print_dpu_info(DPU_TOPO)

#print(json.dumps(DPU_TOPO, indent=1))
