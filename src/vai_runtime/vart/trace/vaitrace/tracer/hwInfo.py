
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

import tracer.tracerBase
import sys
import os
import ctypes
import logging
import json

HW_zynq = [
    {'type': 'CPU', 'file': ['/proc/cpuinfo',
                             '/sys/kernel/debug/clk/acpu/clk_rate']},
    {'type': 'Memory', 'file': ['/proc/meminfo']}
]

HW_alveo = [
    {'type': 'CPU', 'file': ['/proc/cpuinfo']},
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


def getDpuInfo():
    dpuInfo = {'type': 'DPU', 'info': {}}

    try:
        import tools_extra_ops as vart_tools
        dpuInfo["info"] = vart_tools.xdputil_query()
    except:
        pass

    return dpuInfo


def parseText(_type, text, platform):
    cur = dict()
    if _type == 'CPU':
        if platform['machine'] == 'x86_64':
            cur['Cores'] = len([t for t in text if t.startswith('processor')])
            cur['Model'] = "%s" % [t for t in text if t.startswith(
                'model name')][0].split(':')[1].strip()
        else:
            cur['Cores'] = len([t for t in text if t.startswith('processor')])
            cur['Model'] = "ARM-v%s" % [t for t in text if t.startswith(
                'CPU architecture')][0].split(':')[1].strip()

            for l in text:
                if l.find('acpu') >= 0:
                    freq = text[(text.index(l)+1)].strip()
                    cur['Freq'] = "%d MHz" % (int(freq) / 1000 / 1000)
                    break

    if _type == 'Memory':
        totalMem = int([t for t in text if t.startswith('MemTotal')]
                       [0].split(':')[1].strip().split(' ')[0])
        cur['Total Memory'] = "%d MB" % (totalMem / 1000)

    if _type == 'DPU':
        core_id = None

        for t in text:
            ts = t.split(':')
            if len(ts) != 2:
                continue

            key = ts[0].strip()
            value = ts[1].strip()
            if key == 'DPU Core':
                core_id = "%s_%s" % (_type, value)
                cur.setdefault(core_id, {})
            else:
                if core_id == None:
                    continue
                cur[core_id].setdefault(key, value)

    return {'type': _type, 'info': cur}


def getIPsZynq(path=['/sys/devices/platform/amba/amba:zyxclmm_drm/ip_layout',
                     '/sys/devices/platform/axi/axi:zyxclmm_drm/ip_layout',
                     '/sys/devices/platform/amba_pl@0/amba_pl@0:zyxclmm_drm/ip_layout']):
    cus = {}

    for p in path:
        try:
            ip_layout_data = open(p, 'rb').read()
            ips = ip_layout()

            ctypes.memmove(ctypes.pointer(
                ips), ip_layout_data, len(ip_layout_data))

            for i in range(0, ips.m_count):
                ip = ips.m_ip_data[i]
                cu_name = str()

                for c in ip.m_name:
                    cu_name += chr(c)
                cu_type, cu_name = cu_name.strip('\x00').split(':')

                cu_name = cu_name.replace('_xrt_top', '')
                cu_paddr = hex(ip.ip_u1.m_base_address)

                cus[cu_name] = cu_paddr
            if len(cus) > 0:
                break
        except:
            pass

    if len(cus) == 0:
        logging.warning("Cannot open 'zyxclmm_drm/ip_layout'")

    return {'type': 'CUMap', 'info': cus}


def getHwInfo(platform):
    hwInfo = []
    machine = platform.get('machine', "")
    if machine == 'x86_64':
        hwlist = HW_alveo
    elif machine == 'aarch64':
        hwlist = HW_zynq
    else:
        logging.warning("Un-supported platform")
        return hwInfo

    for h in hwlist:
        text = []
        for f in h['file']:
            """Read method for a file"""
            try:
                if os.path.exists(f):
                    text.append("[%s]\n" % f)
                    text += open(f, 'r').readlines()
                else:
                    text += os.popen(f)
            except:
                logging.error("Getting Hardware Info from [%s] failed" % h)

        hwInfo.append(parseText(h['type'], text, platform))

    if machine == 'aarch64':
        hwInfo.append(getIPsZynq())

    hwInfo.append(getDpuInfo())

    return hwInfo


class hwInfoTracer(tracer.tracerBase.Tracer):
    def __init__(self):
        super().__init__('hwInfo', source=[], compatible={
            'machine': ["aarch64", "x86_64"]})

    def prepare(self, options: dict, debug: bool):
        self.platform = options.get('control').get('platform')

    def process(self, data, t_range=[]):
        pass

    def getData(self):
        return getHwInfo(self.platform)


tracer.tracerBase.register(hwInfoTracer())
