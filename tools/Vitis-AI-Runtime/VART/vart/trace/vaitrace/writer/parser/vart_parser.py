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

import time
import math
import logging
import decimal
from writer.parser.timelineUtil import *
from writer.parser.tracepointUtil import *
from writer.parser import xmodel_parser
import writer.parser.statUtil
from writer.parser.unitUtil import *


POINT_PER_SECOND = 100
WINDOW_LENGTH = 0.30


def ips_profile(dpu_timelines):
    """Add FPS data"""
    ips_ret = []
    eventFPS = []

    totalRunCnt = 0

    n_cores = 0

    for dpu in dpu_timelines:
        if dpu.len() == 0:
            continue
        n_cores = n_cores + 1
        for run in dpu.timeline:
            if run.type != 'period':
                continue

            batch = int(run.batch)

            totalRunCnt += batch
            eventFPS.append((run.startTime, batch))

    eventFPS.sort(key=lambda x: x[0])

    def eventFilter(events, start, end):
        ret = []
        for e in events:
            if e[0] > end:
                break
            elif e[0] < start:
                continue
            else:
                ret.append(e)
        return ret

    if len(eventFPS) < 20:
        return []
    else:
        start_t = eventFPS[0][0]
        end_t = eventFPS[-1][0]
        overall_ips = round(totalRunCnt / (end_t - start_t), 2)
        interval = 1 / POINT_PER_SECOND

        time_point = start_t

        while (time_point + WINDOW_LENGTH < end_t):
            eventsInWindow = eventFilter(
                eventFPS, time_point, time_point + WINDOW_LENGTH)
            if len(eventsInWindow) < 3:
                """Skip this point"""
                time_point += interval
                continue
            frames = 0
            for e in eventsInWindow:
                frames += e[1]
            time = WINDOW_LENGTH
            fps = round(frames / time, 1)
            ips_ret.append([round(time_point * 1000, 2), fps])
            time_point += interval

    logging.info("[Summary] DPU cores involved: %d" % n_cores)
    logging.info("[Summary] Total inferences: %d" % totalRunCnt)
    logging.info("[Summary] Overall inferences/second: %.2f" % overall_ips)

    return ips_ret


class dpu_ip():
    def __init__(self):
        self.core_num = 0
        self.cores = []

    def add_core(self, dpu_ip_core):
        self.core_num = self.core_num + 1
        self.cores.append(dpu_ip_core)

    class dpu_ip_core():
        def __init__(self, cu_core_id, cu_device_id, cu_address, cu_name, cu_full_name, cu_fp, cu_batch=1):
            self.core_id = int(cu_core_id)
            self.device_id = int(cu_device_id)
            self.address = cu_address
            self.name = cu_name
            self.full_name = cu_full_name
            self.fp = cu_fp
            self.batch = cu_batch


class DPUEventParser():
    def __init__(self):
        self.vartThreads = {
            # pid: {runtime info}
        }
        self.xmodelInfo = {}
        self.dpu_timelines = []

    def getDpuRuntimeInfo(self, event, key):
        pid = event.pid
        subgraphName = self.vartThreads.get(pid, {}).get(key)
        return "%s" % subgraphName

    def get_info_from_xmodel(self, trace_name, trace_workload_g):
        for subg in self.xmodelInfo:
            info = self.xmodelInfo[subg]
            model_name = info["name"]
            if model_name != trace_name:
                continue
            model_workload = info["workload"]
            #model_workload_g = (int(model_workload) / 1000.0 / 1000.0 / 1000.0)
            model_workload_g = to_Gi(int(model_workload))
            if math.fabs(float(trace_workload_g) - model_workload_g) < 0.02:
                return (info)
        return {}

    def parseDpuRuntimeEvent(self, event: vaiTimelineEvent):
        runtimeEventInfo = event.info
        subgraphName = "subgraph_unknown"
        batchSize = 1
        it = ""
        ot = ""

        pid = event.pid
        if pid not in self.vartThreads.keys():
            self.vartThreads.setdefault(pid, {"subgraph": None})
            self.vartThreads.setdefault(pid, {"batch": 1})

        runtimeEventList = runtimeEventInfo

        for key in runtimeEventList.keys():
            if key == 'subgraph':
                subgraphName = runtimeEventList.get(key, "subgraph")
            elif key == 'batch':
                batchSize = runtimeEventList.get(key, 1)
            elif key == 'op_num':
                op_num = runtimeEventList.get(key, 0)
            elif key == 'workload':
                workload = float(runtimeEventList.get(key, 0))
                workload = round(workload / 1000.0 / 1000.0 / 1000.0, 2)
            elif key == 'hwconuter':
                pass
            elif key == 'it':
                it = runtimeEventList.get(key, None)
                if it.find('(') > 0:
                    it = it.split('(')[0]
            elif key == 'ot':
                ot = runtimeEventList.get(key, None)
                if ot.find('(') > 0:
                    ot = ot.split('(')[0]
            else:
                pass

        self.vartThreads[pid]["subgraph"] = subgraphName
        self.vartThreads[pid]["batch"] = batchSize
        self.vartThreads[pid]["workload"] = workload
        self.vartThreads[pid]["it"] = it
        self.vartThreads[pid]["ot"] = ot

    def create_dpu_ip(self, data):
        self.dpu_ip = dpu_ip()

        dpu_controller_info = select_trace(data, "classname", "dpu-controller")
        dpu_core_num = len(dpu_controller_info)

        for i in dpu_controller_info:
            cu_core_id = int(i.get("cu_core_id", -1))
            cu_device_id = int(i.get("cu_device_id", -1))
            if (cu_core_id == -1) or (cu_device_id == -1):
                logging.error(
                    "Cannot find DPU id in dpu_controller_info, this trace is Error:")
                logging.error("[%s]" % i)
                continue

            cu_addr = i.get("cu_addr", 0)
            cu_name = i.get("cu_name", "DPU_%d" % dpu_controller_info.index(i))
            cu_full_name = i.get("cu_full_name", cu_name)
            cu_fingerprint = i.get("cu_fingerprint", 0)

            """self, cu_core_id, cu_address, cu_name, cu_full_name, cu_fp"""
            core = dpu_ip.dpu_ip_core(
                cu_core_id, cu_device_id, cu_addr, cu_name, cu_full_name, cu_fingerprint)
            self.dpu_ip.add_core(core)

    def parse_trace(self, trace_data):
        """All trace record sort by time"""
        trace_data.sort(key=lambda x: float(x.get('ts')))

        """The first event must be 'dpu-runner'"""
        runtime_env_valid = {}

        for l in trace_data:
            event = tracepointEvent(l).toTimelineEvent()

            """Do Timeline Sync"""
            # TBD

            if event.coreType == "dpu-runner":
                self.parseDpuRuntimeEvent(event)
                runtime_env_valid[event.pid] = True
                continue

            if runtime_env_valid.get(event.pid, False) == False:
                continue

            """ Info get from runtime """
            event.batch = self.getDpuRuntimeInfo(event, "batch")
            event.op_num = 1
            event.subgraph = self.getDpuRuntimeInfo(event, "subgraph")
            event.workload = self.getDpuRuntimeInfo(event, "workload")
            event.it = self.getDpuRuntimeInfo(event, "it")
            event.ot = self.getDpuRuntimeInfo(event, "ot")

            """ Info get from xmodel """
            xmodel_i = self.get_info_from_xmodel(
                event.subgraph, event.workload)
            event.op_num = xmodel_i.get("op_num", event.op_num)

            """ Updata high precision workload """
            event.workload_raw = xmodel_i.get("workload", event.workload)

            event.load_io_img_size = xmodel_i.get("load_io_img_size", 0)
            event.load_io_para_size = xmodel_i.get("load_io_para_size", 0)
            event.save_io_size = xmodel_i.get("save_io_size", 0)
            #event.i_tensor_shape = xmodel_i.get("i_tensor_shape", "")
            #event.o_tensor_shape = xmodel_i.get("o_tensor_shape", "")

            if event.coreType == 'dpu-controller':
                self.dpu_timelines[event.coreId].add(event)

        """ Fix dpu ip [batch size] field """
        for dpu in self.dpu_timelines:
            if dpu.len() == 0:
                continue

            first_event = dpu.timeline[0]
            core_id = first_event.coreId
            batch = int(first_event.batch)
            self.dpu_ip.cores[core_id].batch = batch

    def parse(self, data, options):
        """Two types of event tracing data included: dpuRuntimeEvent & dpuControllerEvent"""
        cuRetData = {}

        """get xmodel info from: {'classname': 'subgraph_info', 'info_file': '/tmp/vaitrace_subgraph_info_15230', 'section': 'INFO'}"""
        subgraph_info = select_trace(data, 'classname', "subgraph_info")
        self.xmodelInfo = xmodel_parser.xmodel_get_info(subgraph_info)

        """Prepare at most 8 DPU timelines"""
        self.dpu_timelines = createTimelines('DPU', 8, options)
        time_offset = options.get("time_offset", 0)
        time_limit = options.get("time_limit", float('inf'))

        data = select_trace_classes(data, ["dpu-runner", "dpu-controller"])
        self.create_dpu_ip(select_trace(data, 'section', "INFO"))
        self.parse_trace(select_trace(data, 'section', "TRACE"))

        return self.dpu_timelines

    def get_dpu_profile_summary(self, fmt="vtf"):
        """
        DPU Profile Summary Format:
        Kernel Name,Number Of Runs,CU Full Name,Minimum Time (ms),Average Time (ms),Maximum Time (ms),Workload(GOP),DPU Performance(GOP/s),Mem IO(MB),Mem Bandwidth(MB/s),
        """

        subGraphStat = {}
        ret = []

        for dpu in self.dpu_timelines:
            if dpu.len() == 0:
                continue
            for run_event in dpu.timeline:
                if run_event.type != 'period':
                    continue

                dpu_id = run_event.coreId
                subgrap_name = run_event.subgraph.strip()
                workload_raw = run_event.workload_raw
                subg_idx = "%s|%s" % (subgrap_name, workload_raw)
                time = run_event.duration

                if (subg_idx in self.xmodelInfo):
                    pass

                subGraphStat.setdefault(dpu_id, {}).setdefault(
                    subg_idx, []).append(time)

        for core_id in subGraphStat.keys():
            for subg_idx in subGraphStat[core_id]:

                subg_name = subg_idx.split('|')[0]
                times = sorted(subGraphStat[core_id][subg_idx])

                min_t = times[0] * 1000
                max_t = times[-1] * 1000
                avg_t = sum(times) / len(times) * 1000
                runs = len(times)

                dpu_core = self.dpu_ip.cores[core_id]
                cu_name = dpu_core.name
                batch = dpu_core.batch
                display_name = "%s:batch-%d" % (cu_name, batch)

                workload = self.xmodelInfo.get(subg_idx, {}).get(
                    "workload", 0)
                #workload = to_Gi(workload)

                perf = workload * batch / avg_t * 1000  # to GOP/ms

                load_img_size = int(self.xmodelInfo.get(
                    subg_idx, {}).get("load_io_img_size", 0))
                load_para_size = int(self.xmodelInfo.get(
                    subg_idx, {}).get("load_io_para_size", 0))
                rank_id = int(self.xmodelInfo.get(
                    subg_idx, {}).get("rank_id", 0))

                write_fm_size = batch * int(self.xmodelInfo.get(
                    subg_idx, {}).get("save_io_size", 0))
                #write_fm_size = to_MB(write_fm_size)

                read_fm_size = load_img_size * batch
                #read_fm_size = to_MB(read_fm_size)

                read_wb_size = load_para_size
                #read_wb_size = to_MB(read_wb_size)

                mem_io = read_fm_size + read_wb_size + write_fm_size
                mem_io_bw = mem_io / avg_t * 1024  # to MB/ms
                #print(subg_name, cu_name, min_t, avg_t, max_t, runs, workload, write_fm_size)

                prt_data_set = {
                    "subg_name": subg_name,
                    "runs": runs,
                    "display_name": display_name,
                    "min_t": uConv(min_t),
                    "avg_t": uConv(avg_t),
                    "max_t": uConv(max_t),
                    "workload": uConv(workload, "Gi"),
                    "perf": uConv(perf, "Gi"),
                    "read_wb_size": uConv(read_wb_size, "MB"),
                    "read_fm_size": uConv(read_fm_size, "MB"),
                    "write_fm_size": uConv(write_fm_size, "MB"),
                    "mem_io_bw": uConv(mem_io_bw, "MB"),
                    "mem_io": uConv(mem_io, "MB"),
                    "rank_id": rank_id
                }

                txt_fmt = "{subg_name},{runs},{display_name},{min_t},{avg_t},{max_t},{workload},{perf},{read_wb_size},{read_fm_size},{write_fm_size},{mem_io_bw},{rank_id},\n"
                vtf_fmt = "{subg_name},{runs},{display_name},{min_t:.3f},{avg_t:.3f},{max_t:.3f},{workload:.3f},{perf:.3f},{mem_io:.3f},{mem_io_bw:.3f},\n"
                if (fmt == "txt"):
                    ret.append(txt_fmt.format(**prt_data_set))
                    # ret.append("%s,%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%d,\n" % (
                    #subg_name, runs, display_name, min_t, avg_t, max_t, workload, perf,
                    # read_wb_size, read_fm_size, write_fm_size, mem_io_bw, rank_id))
                else:
                    ret.append(vtf_fmt.format(**prt_data_set))
                    # ret.append("%s,%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,\n" % (
                    # subg_name, runs, display_name, min_t, avg_t, max_t, workload, perf, mem_io, mem_io_bw))

        return ret

    def get_dpu_ips(self):
        return ips_profile(self.dpu_timelines)
