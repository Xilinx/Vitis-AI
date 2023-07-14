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

from .ascii_table import *
from collections import namedtuple

import sys
import os
import logging

from writer.parser.vart_parser import *
from writer.parser.tracepointUtil import *
from writer.parser.statUtil import *
from writer.parser.pyfunc import *
from writer.parser.cppfunc import *
from writer.parser.cpu_task import *

vtf_events = []
literal_pool = []
dpu_core_num = 0
DEBUG_MODE = False
XRT_INFO = {}
TIMESYNC = {}
dpu_profile_summary = []
cpuTaskSummary = []
cppFuncSummary = []
pyFuncSummary = []
SUBG_DPSP_LEN = 60
OPS_DPSP_LEN = 60

output_f = sys.stdout


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

    dpu_profile_summary = dpu_parser.get_dpu_profile_summary( {"run_machine": runmachine}, "txt")

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
            pass

    vtf_events.sort(key=lambda x: x.ts)


DPU_TABLE_NOTES = \
    """
"~0": Value is close to 0, Within range of (0, 0.001)
Bat: Batch size of the DPU instance
WL(Work Load): Computation workload (MAC indicates two operations), unit is GOP
SW_RT(Software Run time): The execution time calculate by software in milliseconds, unit is ms
HW_RT(Hareware Run time): The execution time from hardware counter in milliseconds, unit is ms
Effic(Efficiency): The DPU actual performance divided by peak theoretical performance,unit is %
LdFM(Load Size of Feature Map): External memory load size of feature map, unit is MB
LdWB(Load Size of Weight and Bias): External memory load size of bias and weight, unit is MB
StFM(Store Size of Feature Map): External memory store size of feature map, unit is MB
AvgBw(Average bandwidth): External memory average bandwidth. unit is MB/s
....
"""

DPU_TABLE_CLOUD_NOTES = \
    """
"~0": Value is close to 0, Within range of (0, 0.001)
Bat: Batch size of the DPU instance
WL(Work Load): Computation workload (MAC indicates two operations), unit is GOP
SW_RT(Software Run time): The execution time calculate by software in milliseconds, unit is ms
LdFM(Load Size of Feature Map): External memory load size of feature map, unit is MB
LdWB(Load Size of Weight and Bias): External memory load size of bias and weight, unit is MB
StFM(Store Size of Feature Map): External memory store size of feature map, unit is MB
AvgBw(Average bandwidth): External memory average bandwidth. unit is MB/s
....
"""

def print_dpu_table(dpu_summary_data, runmachine):
    """
    ['subgraph_conv1,651,DPUCZDX8G_1:batch-1,0.067,11.461,34.191,7.718,673.388,49.947,4357.893,\n',
     'subgraph_conv1,215,DPUCZDX8G_2:batch-1,34.404,34.585,45.104,7.718,223.155,49.947,1444.167,\n']
    """
    #gops = hwin_summary_data[1]
    #freq = hwin_summary_data[0]
    header = ['DPU Id', 'Bat', 'DPU SubGraph',
              'WL', 'SW_RT', 'HW_RT', 'Effic', 'LdWB', 'LdFM', 'StFM', 'AvgBw']

    header_cloud = ['DPU Id', 'Bat', 'DPU SubGraph',
                          'WL', 'SW_RT', 'LdWB', 'LdFM', 'StFM', 'AvgBw']
    pr_data = []
    # pr_data.append(header)

    for row in dpu_summary_data:
        items = row.strip().split(',')
        subgraph_name = items[0].replace("subgraph_", "", 1)
        if len(subgraph_name) > SUBG_DPSP_LEN:
            subgraph_name = "..." + subgraph_name[-SUBG_DPSP_LEN:]

        runs = items[1]

        device_raw = items[2]
        ip_name = device_raw.split(':')[0]
        ip_batch = device_raw.split(':')[1].split('-')[1]

        min_t = items[3]
        ave_t = items[4]
        max_t = items[5]

        workload = items[6]
        effic = items[7]
        mem_ld_fm = items[8]
        mem_ld_w = items[9]
        mem_st_fm = items[10]
        mem_io_bw = items[11]

        hw_rt = items[12]
        rank_id = int(items[13])
        #bandwidth = "{:,}".format(round(float(items[11])))
        if runmachine == "aarch64":
            pr_data.append([ip_name, ip_batch, subgraph_name,
                workload, ave_t, hw_rt, effic, mem_ld_fm, mem_ld_w, mem_st_fm, mem_io_bw, rank_id])
        elif runmachine == "x86_64":
            pr_data.append([ip_name, ip_batch, subgraph_name,
                workload, ave_t, mem_ld_fm, mem_ld_w, mem_st_fm, mem_io_bw, rank_id])

        

    pr_data.sort(key=lambda a: (a.pop() + (hash(a[0]) % 128) * 4096))

    if len(pr_data) == 0:
        return
    if runmachine == "aarch64":
        pr_data.insert(0, header)
    elif runmachine == "x86_64":
        pr_data.insert(0, header_cloud)
    print("DPU Summary:", file=output_f)
    print_ascii_table(pr_data, output_f)
    if runmachine == "aarch64":
        print("\nNotes:%s" % DPU_TABLE_NOTES, file=output_f)
    elif runmachine == "x86_64":
        print("\nNotes:%s" % DPU_TABLE_CLOUD_NOTES, file=output_f)


def print_cpu_task_table(cpu_task_summary):
    if len(cpu_task_summary) == 0:
        return

    header = ['CPU SubGraph', 'OPs', 'Device', 'Runs', 'AverageRunTime(ms)']
    pr_data = []
    pr_data.append(header)

    for row in cpu_task_summary:
        items = row.strip().split(',')

        subgraph_name = items[0].replace("subgraph_", "", 1)
        if len(subgraph_name) > SUBG_DPSP_LEN:
            subgraph_name = "..." + subgraph_name[-SUBG_DPSP_LEN:]
        ops = ""
        dev = "CPU"

        runs = items[1]
        min_t = items[3]
        ave_t = items[4]
        max_t = items[5]
        ops = items[6]
        if (len(ops) > OPS_DPSP_LEN):
            ops = "..." + ops[-OPS_DPSP_LEN:]

        pr_data.append([subgraph_name, ops, dev, runs, ave_t])

    print("\nCPU OPs in Graph(called by GraphRunner):", file=output_f)
    print_ascii_table(pr_data, output_f)


def print_power_info_table(power_info_data):

    header = ['idle power', 'max power', 'ave power']
    pr_data = []
    pr_data.append(header)

    idle_power = power_info_data[0]
    max_power = power_info_data[1]
    ave_power = power_info_data[2]

    pr_data.append([idle_power, max_power, ave_power])

    print("power summary:", file=output_f)
    print_ascii_table(pr_data, output_f)


def print_cpu_func_table(cpp_summary, py_summary):
    """
    ['vitis::ai::ConfigurableDpuTaskImp::setInputImageBGR_1,657,CPU,0.474,0.488,0.652,\n',
     'xir::XrtCu::run,1314,CPU,0.063,7.181,21.457,\n', 'vitis::ai::DpuTaskImp::run,657,CPU,13.180,14.381,21.618,\n']
    """

    header = ['Function', 'Device', 'Runs', 'AverageRunTime(ms)']
    pr_data = []
    pr_data.append(header)

    cpu_func_summary = cpp_summary + py_summary

    if len(cpu_func_summary) == 0:
        return

    for row in cpp_summary:
        items = row.strip().split(',')

        function_name = items[0]
        dev = "CPU"
        runs = items[1]

        min_t = items[3]
        ave_t = items[4]
        max_t = items[5]

        pr_data.append([function_name, dev, runs, ave_t])

    for row in py_summary:
        items = row.strip().split(',')

        function_name = "%s@py" % items[0]
        dev = "CPU"
        runs = items[1]

        min_t = items[3]
        ave_t = items[4]
        max_t = items[5]

        pr_data.append([function_name, dev, runs, ave_t])

    print("\nCPU Functions(Not in Graph, e.g.: pre/post-processing, vai-runtime):", file=output_f)
    print_ascii_table(pr_data, output_f)


# Statistical information for DPU kernels(min/max time etc.)
def output_profile_summary(runmachine):
    global pyFuncSummary, cpuTaskSummary, cppFuncSummary, dpu_profile_summary

    print_dpu_table(dpu_profile_summary, runmachine)
    print_cpu_task_table(cpuTaskSummary)
    print_cpu_func_table(cppFuncSummary, pyFuncSummary)


def output(runmachine, saveTo=None):
    global output_f

    if saveTo != None:
        logging.info("Saving report to: %s" % saveTo)
        try:
            output_f = open(saveTo, "w+t")
        except:
            output_f = sys.stdout
            logging.error("Fail opening: %s" % saveTo)

    output_profile_summary(runmachine)


def xat_to_txt(xat, options):
    global XRT_INFO, DEBUG_MODE, cpuTaskSummary, cppFuncSummary, pyFuncSummary

    saveTo = options.get('cmdline_args', {}).get('output', None)
    runmachine = options.get('control', {}).get('platform',{}).get('machine',{})

    convert_dpu(xat.get('vart'), xat.get('hwInfo'), options)
    cpuTaskSummary = convert_cpu_task(xat.get('vart', {}))
    pyFuncSummary = convert_pyfunc(xat.get('pyfunc', {}))
    cppFuncSummary = convert_cppfunc(xat.get('function', {}))

    output(runmachine,saveTo)
