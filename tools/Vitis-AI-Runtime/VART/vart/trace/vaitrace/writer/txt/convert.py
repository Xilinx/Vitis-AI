# Copyright 2019 Xilinx Inc.

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

vtf_events = []
literal_pool = []
dpu_core_num = 0
DEBUG_MODE = False
XRT_INFO = {}
TIMESYNC = {}
dpu_profile_summary = []
cppFuncSummary = []
pyFuncSummary = []

output_f = sys.stdout

def convert_dpu(raw_data):
    global literal_pool, dpu_core_num, vtf_events, dpu_profile_summary
    global TIMESYNC
    offset = TIMESYNC.get('sync', {}).get('vart')
    timeout = TIMESYNC.get('timeout', 0)

    dpu_parser = DPUEventParser()
    timelines = dpu_parser.parse(
        raw_data, {"time_offset": offset, "time_limit": timeout})

    dpu_profile_summary = dpu_parser.get_dpu_profile_summary()

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


def print_table(input):
    header = ['Device', 'SubGraph',
              'Workload(GOP)', 'AverageRunTime(ms)', 'Perf(GOP/s)', 'Mem(MB)', 'MB/s']
    data = []

    for d in input.keys():
        n = input[d]['name'].split(':')[0]
        dpu_id = "%s:%s" % (n, d)
        for subg_name in input[d]['subgraphs'].keys():
            workload = input[d]['subgraphs'][subg_name]['workload'].split()[0]
            run_t = input[d]['subgraphs'][subg_name]['ave_t']
            perf = input[d]['subgraphs'][subg_name]['effic'].split()[0]
            mem_r = input[d]['subgraphs'][subg_name].get('mem_r', 0)
            mem_w = input[d]['subgraphs'][subg_name].get('mem_w', 0)
            mb_s = "%.2f" % ((int(mem_r + int(mem_w))) /
                             1024 / 1024 / float(run_t) * 1000)
            mem = "%.2f" % ((int(mem_r) + int(mem_w)) / 1024 / 1024)

            data.append([dpu_id, subg_name, workload, run_t, perf, mem, mb_s])

    print_ascii_table([header, *data], output_f)


def print_summary():
    pr_info = DPU_INFO

    for id in pr_info.keys():
        """remove items"""
        pr_info[id].pop('first_event_time')
        pr_info[id].pop('last_event_time')
        pr_info[id].pop('idle_time')

        """format items"""
        pr_info[id]['util'] = "%.2f %%" % pr_info[id]['util']

        for subg_name in pr_info[id]['subgraphs'].keys():
            subg = pr_info[id]['subgraphs'][subg_name]
            subg['min_t'] = round(subg['min_t'], 3)
            subg['max_t'] = round(subg['max_t'], 3)
            subg['ave_t'] = round(subg['ave_t'], 3)
            subg['workload'] = "%.2f GOP" % (float(subg['workload']))
            subg['effic'] = "%.2f GOP/s" % (subg['effic'])

    print_table(pr_info)


def print_dpu_table(dpu_summary_data):
    """
    ['subgraph_conv1,651,DPUCZDX8G_1:batch-1,0.067,11.461,34.191,7.718,673.388,49.947,4357.893,\n',
     'subgraph_conv1,215,DPUCZDX8G_2:batch-1,34.404,34.585,45.104,7.718,223.155,49.947,1444.167,\n']
    """
    header = ['DPU', 'Batch', 'SubGraph',
              'Workload(GOP)', 'RunTime(ms)', 'Perf(GOP/s)', 'Mem(MB)', 'MB/s']
    pr_data = []
    pr_data.append(header)

    for row in dpu_summary_data:
        items = row.strip().split(',')
        subgraph_name = items[0]
        if len(subgraph_name) > 65:
            subgraph_name = subgraph_name[0:65] + "..."

        runs = items[1]

        device_raw = items[2]
        ip_name = device_raw.split(':')[0]
        ip_batch = device_raw.split(':')[1].split('-')[1]

        min_t = items[3]
        ave_t = items[4]
        max_t = items[5]

        workload = items[6]
        effic = items[7]
        mem = items[8]
        bandwidth = items[9]

        pr_data.append([ip_name, ip_batch, subgraph_name,
                        workload, ave_t, effic, mem, bandwidth])

    print("DPU Summary:", file=output_f)
    print_ascii_table(pr_data, output_f)


def print_cpu_table(cpp_summary, py_summary):
    """
    ['vitis::ai::ConfigurableDpuTaskImp::setInputImageBGR_1,657,CPU,0.474,0.488,0.652,\n',
     'xir::XrtCu::run,1314,CPU,0.063,7.181,21.457,\n', 'vitis::ai::DpuTaskImp::run,657,CPU,13.180,14.381,21.618,\n']
    """

    header = ['Function', 'Device', 'Runs', 'AverageRunTime(ms)']
    pr_data = []
    pr_data.append(header)

    for row in cpp_summary:
        items = row.strip().split(',')

        function_name = items[0]
        dev = "CPU"
        runs = items[1]

        min_t = items[3]
        ave_t = items[4]
        max_t = items[5]

        pr_data.append([function_name, dev, runs, ave_t])

    print("\nCPU Summary:", file=output_f)
    print_ascii_table(pr_data, output_f)


# Statistical information for DPU kernels(min/max time etc.)
def output_profile_summary():
    global pyFuncSummary, cppFuncSummary, dpu_profile_summary

    print_dpu_table(dpu_profile_summary)
    print_cpu_table(cppFuncSummary, pyFuncSummary)


def output(saveTo=None):
    global output_f

    if saveTo != None:
        logging.info("Saving report to: %s" % saveTo)
        try:
            output_f = open(saveTo, "w+t")
        except:
            output_f = sys.stdout
            logging.error("Fail opening: %s" % saveTo)

    output_profile_summary()


def xat_to_txt(xat, saveTo=None):
    global XRT_INFO, DEBUG_MODE, cppFuncSummary, pyFuncSummary

    convert_dpu(xat.get('vart'))
    pyFuncSummary = convert_pyfunc(xat.get('pyfunc', {}))
    cppFuncSummary = convert_cppfunc(xat.get('function', {}))

    output(saveTo)
