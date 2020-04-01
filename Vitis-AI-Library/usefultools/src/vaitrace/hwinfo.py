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

import sys, os, json, ctypes

HWs = [
        {'type': 'CPU', 'file': ['/proc/cpuinfo', '/sys/kernel/debug/clk/acpu/clk_rate']},
        {'type': 'Memory', 'file': ['/proc/meminfo']}
      ]

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

MEM_TYPE = [
    "MEM_DDR3",
    "MEM_DDR4",
    "MEM_DRAM",
    "MEM_STREAMING",
    "MEM_PREALLOCATED_GLOB",
    "MEM_ARE",
    "MEM_HBM",
    "MEM_BRAM",
    "MEM_URAM",
    "MEM_STREAMING_CONNECTION "
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
        ("m_connection", connection*64)
    ]

"""   IP_LAYOUT SECTION   """
class IP_CONTROL:
    AP_CTRL_HS    = 0
    AP_CTRL_CHAIN = 1
    AP_CTRL_NONE  = 2
    AP_CTRL_ME    = 3

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


def parseText(_type, text):
    cur = dict()
    if _type == 'CPU':
        cur['Cores'] = len([t for t in text if t.startswith('processor')])
        cur['Model'] = "ARM-v%s" % [t for t in text if t.startswith('CPU architecture')][0].split(':')[1].strip()

        for l in text:
            if l.find('acpu') >= 0:
                freq = text[(text.index(l)+1)].strip()
                cur['Freq'] = "%d MHz" % (int(freq) / 1000 / 1000)
                break

    if _type == 'Memory':
        totalMem = int([t for t in text if t.startswith('MemTotal')][0].split(':')[1].strip().split(' ')[0])
        cur['Total Memory'] = "%d MB" % (totalMem / 1024)
    
    return {'type': _type, 'info': cur}

def getIPs(path = '/sys/devices/platform/amba/amba:zyxclmm_drm/ip_layout'):
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

        cus[cu_name] = cu_paddr

    return {'type': 'CUs', 'info': cus}

def getMemTopology(path = '/sys/devices/platform/amba/amba:zyxclmm_drm/mem_topology'):
    mem_info = []
    mem_topology_data = open(path, 'rb').read()
    mem = mem_data()

    ctypes.memmove(ctypes.pointer(mem), mem_topology_data, len(mem_topology_data))

    for i in range(0, mem.m_count):
        m = mem.m_mem_data[i]

        m_tag = str()
        for c in m.m_tag:
            m_tag += chr(c)

        m_type = MEM_TYPE[m.m_type]
        m_base_address = m.m_base_address
        m_size = m.m_size

        meminfo.append({
            'tag': m_tag,
            'type': m_type,
            'base_address': m_base_address,
            'size': m_size})

    return {'type': 'MEMs', 'info': meminfo}

def getHwInfo():
    hwInfo = []
    for h in HWs:
        text = []
        for f in h['file']:
            """Read method for a file"""
            if os.path.exists(f):
                text.append("[%s]\n" % f)
                text += open(f, 'r').readlines()
            else:
                text += os.popen(f)

        hwInfo.append(parseText(h['type'], text))

    hwInfo.append(getIPs())
    
    return hwInfo


#print("#HWINFO " + json.dumps(getHwInfo()))
