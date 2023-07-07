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

import sys
import os
import logging
import json
import uuid

from vaitraceOptions import merge

from writer.parser.vart_parser import *
from writer.parser.tracepointUtil import *
from writer.parser.statUtil import *
from writer.parser.pyfunc import *
from writer.parser.cppfunc import *
from writer.parser.cpu_task import *
from writer.parser.power import *

from writer.parser.sched_parser.sched_parser import sched_parser_main

vtf_events = []
dpu_ip_summary = {}


def convert_dpu(raw_data, dpu_ip_data, options):
    global dpu_ip_summary

    runmachine = options.get('control', {}).get('platform',{}).get('machine',{})
    dpu_parser = DPUEventParser()
    timelines = dpu_parser.parse(raw_data, dpu_ip_data, {})

    dpu_ip_summary = {"dpu": {"ip": dpu_parser.get_dpu_ip_summary()}}
    report = {
        "dpu": {
            "profiling": {},
            "scheduling": {}
        }
    }

    for timeline in timelines:
        if timeline.coreType == "DPU":
            cu_name = timeline.get_core_name()
            scheduling_rate = timeline.get_util()
            if cu_name == None or scheduling_rate == -1:
                continue

            # print(f"{dpu_name}: {scheduling_rate}")
            report["dpu"]["scheduling"].setdefault(cu_name, 0)
            report["dpu"]["scheduling"][cu_name] = scheduling_rate

    # "{subg_name},{runs},{cu_name},{batch},{min_t},{avg_t},{max_t},{workload},{effic},{read_wb_size},{read_fm_size},{write_fm_size},{mem_io_bw},{hw_rt},{depth},\n"
    dpu_summary = dpu_parser.get_dpu_profile_summary({"run_machine": runmachine},"json")

    for row in dpu_summary:
        row = row.strip().split(',')
        dpu_model_profiling_item = {
            "subg_name":        str(row[0]),
            "runs":             int(row[1]),
            "cu_name":          str(row[2]),
            "batch":            int(row[3]),
            "min_time":         float(row[4]),
            "avg_time":         float(row[5]),
            "max_time":         float(row[6]),
            "workload":         int(row[7]),
            "efficiency":       float(row[8]),
            "read_weight_bias_size":    int(row[9]),
            "read_feature_map_size":    int(row[10]),
            "write_featyre_map_size":   int(row[11]),
            "mem_io_bw":        float(row[12]),
            "hardware_time":    float(row[13]),
            "depth":     int(row[14]),
            "input_tensor_shape": str(row[15].replace("_", ",")),
            "output_tensor_shape": str(row[16].replace("_", ","))
        }

        # sort by cu
        cu_name = dpu_model_profiling_item["cu_name"]
        report["dpu"]["profiling"].setdefault(cu_name, [])
        report["dpu"]["profiling"][cu_name].append(dpu_model_profiling_item)

    return report


def convert_power_info(raw_data):
    return analyse_power(raw_data)


def get_power_info(power_info_data):
    peak = power_info_data.get("peak_power", 0)
    idle = power_info_data.get("idle_power", 0)
    average = power_info_data.get("ave_power", 0)
    limit = power_info_data.get("limit_power", -1)

    return {"board": {
        "power.peak": peak,
        "power.idle": idle,
        "power.average": average,
        "power.limit": limit
    }
    }


def get_cpu_task_info(cpu_task_summary):
    report = {
        "cpu": {
            "graph_runner.cpu_tasks": [],
            "onnx_runner.cpu_tasks": []
        }
    }

    for row in cpu_task_summary:
        items = row.strip().split(',')

        subgraph_name = items[0].replace("subgraph_", "", 1)

        runs = int(items[1])
        min_t = float(items[3])
        ave_t = float(items[4])
        max_t = float(items[5])
        ops = items[6]

        report_item = {
            "subgraph": subgraph_name,
            "ops": ops,
            "average_time": ave_t,
            "min_time": min_t,
            "max_time": max_t,
            "runs": runs
        }

        report["cpu"]["graph_runner.cpu_tasks"].append(report_item)

    return report


def get_cpu_func_info(cpp_summary, py_summary):
    report = {
        "cpu": {
            "vitis_ai_libaray.cpu_functions": []
        }
    }

    for row in cpp_summary + py_summary:
        items = row.strip().split(',')

        if row in py_summary:
            function_name = "%s@py" % items[0]
        else:
            function_name = items[0]
        runs = int(items[1])

        min_t = float(items[3])
        ave_t = float(items[4])
        max_t = float(items[5])

        report_item = {
            "func": function_name,
            "average_time": ave_t,
            "min_time": min_t,
            "max_time": max_t,
            "runs": runs
        }

        report["cpu"]["vitis_ai_libaray.cpu_functions"].append(report_item)

    return report


def get_cmd_info(cmd_data):
    report = {
        "vaitrace_environment": {}
    }

    report["vaitrace_environment"]["cmd"] = cmd_data.get("cmd", [])
    report["vaitrace_environment"]["platform"] = cmd_data.get("platform", [])
    report["vaitrace_environment"]["time"] = cmd_data.get("time", [])

    return report


def xat_to_json(xat, options):
    global dpu_ip_summary

    saveTo = options.get('cmdline_args', {}).get('output', None)

    dpuProfilingSummary = convert_dpu(xat.get('vart'), xat.get('hwInfo'), options)

    powerInfoSummary = convert_power_info(xat.get('power', {}))
    cpuTaskSummary = convert_cpu_task(xat.get('vart', {}))
    pyFuncSummary = convert_pyfunc(xat.get('pyfunc', {}))
    cppFuncSummary = convert_cppfunc(xat.get('function', {}))

    cpuUtilSummary = sched_parser_main(xat.get('sched', {}))

    output_f = sys.stdout
    report = {"uuid": uuid.uuid1().hex}

    if saveTo != None:
        logging.info("Saving report to: %s" % saveTo)
        try:
            output_f = open(saveTo, "w+t")
        except:
            output_f = sys.stdout
            logging.error("Opening file failed: %s" % saveTo)

    merge(report, dpuProfilingSummary)
    merge(report, get_power_info(powerInfoSummary))
    merge(report, get_cpu_task_info(cpuTaskSummary))
    merge(report, get_cpu_func_info(cppFuncSummary, pyFuncSummary))
    merge(report, get_cmd_info(xat.get('cmd', {})))
    merge(report, dpu_ip_summary)
    merge(report, cpuUtilSummary)

    print(json.dumps(report, indent=4, separators=(",", ":")), file=output_f)
