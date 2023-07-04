
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

import csv
import time
import pickle
import sys
import os
import gzip
import logging
from writer.parser.vart_parser import *
from writer.parser.tracepointUtil import *
from writer.parser.statUtil import *
from writer.parser.pyfunc import *
from writer.parser.cppfunc import *

event_id = 1
start_id = 1
vtf_events = []
apm_events = []
nmu_events = []
nmu_info = {}
literal_pool = []
dpu_core_num = 0
PID = 0
GEN_TIME = 0
DEBUG_MODE = False
XRT_INFO = {}
TIMESYNC = {}
dpu_profile_summary = []
cppFuncSummary = []
pyFuncSummary = []

"""
EventID:uint64, StartID:uint64, Timestamp:double, bucketID:uint32, Type:int32, [, TYPE SPECIFIC FIELDS]

DPU EVENTS:
DPU_RUN -> EventID, StartID, Timestamp, bucketID:uint32, DPU_RUN, kernelName:MID, ThreadID:uint64, batch_size:uint32, input_tensor:MID, output_tensor:MID 
"""


class VART_VTF:
    def __init__(self, eid, sid, ts, bid, kn, tid, bs, it="", ot="", wl=0, op_num=0, ef=0):
        self.eid = int(eid)
        self.sid = int(sid)

        def time_to_ms(t_s):
            return t_s * 1000
        self.ts = float(time_to_ms(ts))
        self.bid = int(bid)
        self.kn = int(kn)
        self.tid = int(tid)
        self.bs = int(bs)
        self.it = it
        self.ot = ot
        self.wl = round(wl, 3)
        self.ef = round(ef, 2)
        self.opn = op_num
        self.rs = ""

    def __str__(self):
        mandatory = "%d,%d,%f,%d,VART_RUNNER,%d,%d,%d" % (
            self.eid, self.sid, self.ts, self.bid, self.kn, self.tid, self.bs)
        optional = str(self.wl) + ',' + str(self.ef) + ',' + str(self.opn) + \
            ',' + str(self.it) + ',' + str(self.ot) + ',' + str(self.rs) + ','
        return mandatory + ',' + optional + '\n'


def toVTF(e):
    global event_id

    bid = e.coreId + 1
    subgraph_index = literal_pool.index(e.subgraph)

    it_index = literal_pool.index(e.it)
    ot_index = literal_pool.index(e.ot)

    op_num = e.op_num

    """Unit of workload: GOPS"""
    workload = float(e.workload)

    """Unit of efficiency: GOPS/s"""
    time_dur = e.endTime - e.startTime
    efficiency = round(workload / time_dur, 2)

    vtf_event_satat = VART_VTF(
        event_id, 0, e.startTime, bid, subgraph_index, e.pid, e.batch, it_index, ot_index, workload, op_num, efficiency)
    start_id = event_id
    event_id = event_id + 1
    vtf_event_end = VART_VTF(event_id, start_id, e.endTime,
                             bid, subgraph_index, e.pid, e.batch, it_index, ot_index, workload, op_num, efficiency)
    event_id = event_id + 1

    return (vtf_event_satat, vtf_event_end)


def do_timeline_sync(xat, sync_base="xrt"):
    global TIMESYNC
    timesync = xat.get('timesync')
    timesync.pop("time_range")
    timeout = xat.get('cmd', {}).get('timeout')

    timeline_base = float('inf')
    for k in timesync.keys():
        if timesync[k] == 0:
            continue
        if timesync[k] < timeline_base:
            timeline_base = timesync[k]

    for k in timesync.keys():
        if timesync[k] == 0:
            continue
        timesync[k] -= timeline_base

    TIMESYNC = {"sync": timesync, "timeout": timeout}


def convert_dpu(raw_data, hwInfo, options):
    global literal_pool, dpu_core_num, vtf_events, dpu_profile_summary
    global TIMESYNC
    offset = TIMESYNC.get('sync', {}).get('vart')
    timeout = TIMESYNC.get('timeout', 0)

    runmode = options.get('control', {}).get('runmode')

    runmachine = options.get('control', {}).get('platform',{}).get('machine',{})
    dpu_parser = DPUEventParser()
    timelines = dpu_parser.parse(
        raw_data, hwInfo, {"time_offset": offset, "time_limit": timeout, "runmode": runmode})

    dpu_profile_summary = dpu_parser.get_dpu_profile_summary({"run_machine": runmachine})

    """extracting all strings"""
    for dpu in timelines:
        for event in dpu.timeline:
            literal_pool.append(event.subgraph)
            literal_pool.append(event.it)
            literal_pool.append(event.ot)
    literal_pool = list(set(literal_pool))
    literal_pool.insert(0, "RESERVED")

    for dpu_timeline in timelines:
        if len(dpu_timeline.timeline) == 0:
            continue
        dpu_core_num += 1
        for event in dpu_timeline.timeline:
            s, e = toVTF(event)
            vtf_events.append(s)
            vtf_events.append(e)

    vtf_events.sort(key=lambda x: x.ts)


def convert_xapm(apm_data, time_offset=0.0):
    global apm_events

    def toMB_s(r):
        return float(r) / 1000 / 1000

    for rr in apm_data:
        if not rr.startswith("APM "):
            continue
        r = rr.split()[1:]
        timestamp = round((float(r[0]) - time_offset) * 1000, 4)
        if timestamp < 0:
            continue
        s_1_r = round(toMB_s(r[1]), 2)
        s_1_w = round(toMB_s(r[2]), 2)
        s_2_r = round(toMB_s(r[3]), 2)
        s_2_w = round(toMB_s(r[4]), 2)
        s_3_r = round(toMB_s(r[5]), 2)
        s_3_w = round(toMB_s(r[6]), 2)
        s_4_r = round(toMB_s(r[7]), 2)
        s_4_w = round(toMB_s(r[8]), 2)
        s_5_r = round(toMB_s(r[9]), 2)
        s_5_w = round(toMB_s(r[10]), 2)
        apm_events.append([timestamp, s_1_r, s_1_w, s_2_r,
                           s_2_w, s_3_r, s_3_w, s_4_r, s_4_w, s_5_r, s_5_w])

def convert_nmu(nmu_data,options, time_offset=0.0):
    global nmu_events, nmu_info

    if options.get('noc_nmu', '') != True:
        return
    sample_data = []
    def toMB_s(r):
        return float(r) / 1000 / 1000

    nmu_info = nmu_data[0]
    sample_data = nmu_data[1]
    for rr in sample_data:
        if not rr.startswith("NMU "):
            continue
        r = rr.split()[1:]
        timestamp = round((float(r[0]) - time_offset) * 1000, 4)
        if timestamp < 0:
            continue
        s_1_r = round(toMB_s(r[1]), 2)
        s_1_w = round(toMB_s(r[2]), 2)
        s_2_r = round(toMB_s(r[3]), 2)
        s_2_w = round(toMB_s(r[4]), 2)
        s_3_r = round(toMB_s(r[5]), 2)
        s_3_w = round(toMB_s(r[6]), 2)
        s_4_r = round(toMB_s(r[7]), 2)
        s_4_w = round(toMB_s(r[8]), 2)
        s_5_r = round(toMB_s(r[9]), 2)
        s_5_w = round(toMB_s(r[10]), 2)
        nmu_events.append([timestamp, s_1_r, s_1_w, s_2_r,
                           s_2_w, s_3_r, s_3_w, s_4_r, s_4_w, s_5_r, s_5_w])


def output_nmu_trace(OUTPUT_PATH, options):
    if options.get('noc_nmu', False) != True:
        return
    port_name = []
    per_name = []
    f_name = "noc_nmu_trace.csv"
    npi_freq = nmu_info.get('npi_freq', 0)
    sample_interval = nmu_info.get('interval', 0)
    with open(os.path.join(OUTPUT_PATH, f_name), "w+t") as vtf:
        """HEADER"""
        vtf.writelines("NOC_NMU_INFO\n")
        test_info = [
            "interval,%s\n" % sample_interval,
            "npi_freq,%s\n" % npi_freq,
        ]
        vtf.writelines(test_info)
        vtf.writelines("\n")
        sample_nodes = nmu_info.get('nodes', [])
        for p in sample_nodes:
            if p.get('type',' ') == 'nmu':
                port_name.append(p.get('location_name', ''))
        title = "NOC_NMU Bandwith\n"
        vtf.write(title)
        for i in port_name:
            per_name.append(i + "_R")
            per_name.append(i + "_W")
        port_header = "sampletime," + ",".join(per_name) + ",\n"
        vtf.write(port_header)
        for d in nmu_events:
            item = "%.2f,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,\n" % (
                d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10])
            vtf.write(item)
        vtf.write('\n')

def output_vart_trace(OUTPUT_PATH):
    global PID, GEN_TIME, XRT_INFO
    xrt_version = XRT_INFO.get("runtime", {}).get(
        "build", {}).get("version", "0.0.0")
    cus = XRT_INFO.get("board", {}).get("compute_unit", {})
    f_name = "vart_trace.csv"

    with open(os.path.join(OUTPUT_PATH, f_name), "w+t") as vtf:
        """HEADER"""
        vtf.writelines("HEADER\n")
        header = [
            "VTF File Version,1.0\n",
            "VTF File Type,0\n",
            "PID,%d\n" % PID,
            "Generated on,%s\n" % GEN_TIME,
            "Resolution,ms\n",
            "Min Resolution,us\n",
            "Trace Version,1.0\n",
            "XRT Version,%s\n" % xrt_version,
            "Tool Version,2020.1\n"
        ]
        vtf.writelines(header)
        vtf.writelines("\n")

        """STRUCTURE"""
        vtf.writelines("STRUCTURE\n")
        s = []
        s.append("Group_Start,DPU\n")
        for i in range(dpu_core_num):
            try:
                cu_name = cus[str(i)].get("name", "DPU_%d" % i)
            except:
                cu_name = "DPU_%d" % i
            s.append("Dynamic_Row, %d, %s\n" %
                     (i+1, cu_name))

        s.append("Group_End,DPU\n")
        s.append("\n")
        vtf.writelines(s)

        """MAPPING"""
        vtf.writelines("MAPPING\n")
        for item in literal_pool[1:]:
            mapping = "%d,%s\n" % (literal_pool.index(item), item)
            vtf.writelines(mapping)
        vtf.writelines("\n")

        """EVENTS"""
        vtf.writelines("EVENTS\n")
        for e in vtf_events:
            vtf.writelines(str(e))

    logging.info("[%s]: Events number: %d" % (f_name, len(vtf_events)))

# Statistical information for DPU kernels(min/max time etc.)


def output_profile_summary(OUTPUT_PATH):
    f_name = "profile_summary.csv"
    xrt_version = XRT_INFO.get("runtime", {}).get(
        "build", {}).get("version", "0.0.0").strip()
    xrt_branch = XRT_INFO.get("runtime", {}).get(
        "build", {}).get("branch", "N/A").strip()
    xrt_hash = XRT_INFO.get("runtime", {}).get(
        "build", {}).get("hash", "N/A").strip()
    xrt_date = XRT_INFO.get("runtime", {}).get(
        "build", {}).get("date", "N/A").strip()
    xrt_device = XRT_INFO.get("board", {}).get(
        "info", {}).get("dsa_name", "N/A").strip()
    cus = XRT_INFO.get("board", {}).get("compute_unit", {})
    global dpuLatencyStat, GEN_TIME, pyFuncSummary, cppFuncSummary

    with open(os.path.join(OUTPUT_PATH, f_name), "w+t") as csv_f:
        t = GEN_TIME
        HEADER = "Profile Summary\nGenerated on: %s\n" % t +\
                 "XRT build version: %s\n" % xrt_version +\
                 "Build version branch: %s\n" % xrt_branch +\
                 "Build version hash: %s\n" % xrt_hash +\
                 "Build version date: %s\n" % xrt_date +\
                 "Target devices: %s\n\n" % xrt_device
        csv_f.write(HEADER)

        title = "DPU Summary\n"
        column_headers = "Kernel Name,Number Of Runs,CU Full Name,Minimum Time (ms),Average Time (ms),Maximum Time (ms),Workload(GOP),DPU Performance(GOP/s),Mem IO(MB),Mem Bandwidth(MB/s),\n"
        csv_f.write(title)
        csv_f.write(column_headers)

        csv_f.writelines(dpu_profile_summary)

        csv_f.writelines(cppFuncSummary)
        csv_f.writelines(pyFuncSummary)
        csv_f.write('\n')


# DDR bandwidth and throughtput
def output_vitis_ai_profile(OUTPUT_PATH):
    f_name = "vitis_ai_profile.csv"
    global apm_events, DEBUG_MODE

    with open(os.path.join(OUTPUT_PATH, f_name), "w+t") as csv_f:
        title = "DPU Throughput\n"
        column_headers = "timestamp,FPS\n"
        csv_f.write(title)
        csv_f.write(column_headers)

        csv_f.write('\n')
        title = "DDR Bandwidth\n"
        column_headers = "timestamp,DDRC_PORT_S1_Read,DDRC_PORT_S1_Write,DDRC_PORT_S2_Read,DDRC_PORT_S2_Write,DDRC_PORT_S3_Read,DDRC_PORT_S3_Write,DDRC_PORT_S4_Read,DDRC_PORT_S4_Write,DDRC_PORT_S5_Read,DDRC_PORT_S5_Write,\n"
        csv_f.write(title)
        csv_f.write(column_headers)
        for d in apm_events:
            item = "%.2f,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,\n" % (
                d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10])
            csv_f.write(item)
        csv_f.write('\n')


def output(OUTPUT_PATH, options):
    output_vart_trace(OUTPUT_PATH)
    output_nmu_trace(OUTPUT_PATH, options)
    output_profile_summary(OUTPUT_PATH)
    output_vitis_ai_profile(OUTPUT_PATH)


def xat_to_vtf(xat, options):
    saveTo = "./"
    global PID, GEN_TIME, XRT_INFO, DEBUG_MODE, cppFuncSummary, pyFuncSummary
    PID = xat.get('cmd', {}).get('pid', 0)
    GEN_TIME = xat.get('cmd', {}).get('time', "0000-00-00 00:00:00")
    XRT_INFO = xat.get('xrt', {})
    runmode = xat.get('cmd', {}).get('runmode', "")
    if runmode == "debug":
        DEBUG_MODE = True

    xapm_ts_offset = xat.get('timesync', {}).get('vart', 0.0)
    do_timeline_sync(xat)
    convert_dpu(xat.get('vart'), xat.get('hwInfo'), options)
    convert_xapm(xat.get('xapm', {}), xapm_ts_offset)
    #convert_nmu(xat.get('nmu', {}), options, xapm_ts_offset)
    pyFuncSummary = convert_pyfunc(xat.get('pyfunc', {}))
    cppFuncSummary = convert_cppfunc(xat.get('function', {}))
    output(saveTo, options)
