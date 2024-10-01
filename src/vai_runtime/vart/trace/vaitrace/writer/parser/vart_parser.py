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

import time
import os
import math
import logging
import decimal
import json
import vaitraceSetting
from copy import deepcopy
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

        # from [xdputil query]
        self.ip_version = None
        self.timestamp = None
        self.commit_id = None

    def __str__(self):
        return json.dumps(self.summary(), sort_keys=True, indent=4, separators=(',', ': '))

    def summary(self):
        ret = deepcopy(self.__dict__)

        ret["cores"] = []

        for c in self.cores:
            ret["cores"].append(deepcopy(c.__dict__))

        return ret

    def add_core(self, new_core):
        if self.check_unique(new_core.full_name):
            self.core_num = self.core_num + 1
            self.cores.append(new_core)

    def check_unique(self, cu_full_name):
        for c in self.cores:
            if cu_full_name == c.full_name:
                return False

        return True

    class dpu_ip_core():
        def __init__(self, cu_core_id, cu_device_id, cu_address, cu_name, cu_full_name, cu_fp, cu_hw_rt=0, cu_batch=1):
            self.core_id = int(cu_core_id)
            self.device_id = int(cu_device_id)
            self.address = cu_address
            self.name = cu_name
            self.full_name = cu_full_name
            self.fp = hex(int(cu_fp))
            self.batch = cu_batch

            # from [xdputil query]
            self.axilite_freq = None
            self.isa = None
            self.pl_freq = None
            self.aie_freq = None
            self.peak_ops = None
            self.arch = None
            self.type = None
            self.load_parallel = None
            self.save_parallel = None


class DPUEventParser():
    def __init__(self):
        self.vartThreads = {
            # pid: {runtime info}
        }
        self.xmodelInfo = {}
        self.dpu_timelines = []
        self.dpuIpInfo = []

    def getDpuAieClk(self, dpu_arch):

        idcode = '0x' + vaitraceSetting.getChipId().get('idcode',{})[-7:]
        mepll_base_addr_dict = {"0x4ca8093": 0xf70a0000, "0x4cd3093": 0xf6d10000,"0x4cc8093": 0xf6420000,
                "0x4cc0093": 0xf6300000, "0x4cd0093": 0xf6d10000, "0x4c98093": 0xf6d10000, "0x4c9b093": 0xf6d10000}
        mepll_base_addr =  mepll_base_addr_dict.get(idcode, 0)
        if mepll_base_addr == 0 :
            logging.warning("Not support %s aie clk " % hostname)
            return 1
        MEPLL_CTRL = mepll_base_addr + 0x100
        ME_CORE_REF_CTRL = mepll_base_addr + 0x138
        PMCPLL_CTRL = 0xF1260040
        NOCPLL_CTRL = 0xF1260050
        HSM0_REF_CTRL = 0xF1260148
        
        ref_clk = int(os.popen('cat /sys/kernel/debug/clk/ref_clk/clk_rate').read())
        hsm0_ref = int(os.popen(f"devmem {HSM0_REF_CTRL} 32").read(), 16)

        if (hsm0_ref & 0x7) == 0:
            n_p_pll = int(os.popen(f"devmem {PMCPLL_CTRL} 32").read(), 16)
        elif (hsm0_ref & 0x7) == 3:
            n_p_pll = int(os.popen(f"devmem {NOCPLL_CTRL} 32").read(), 16)
        else:
            print("get a error value from hsm0")

        clkoutdiv = math.pow(2, ((n_p_pll & 0x30000) >> 16))
        n_p_cla = (n_p_pll & 0xFF00) >> 8  # bit 15:8
        n_p_clk = ref_clk * n_p_cla / clkoutdiv
        hsm0_cal = (hsm0_ref & 0x3ff00) >> 8  # bit 17:8
        hsm0_clk = n_p_clk / hsm0_cal
        mepll = int(os.popen(f"devmem {MEPLL_CTRL} 32").read(), 16)
        mepll_cal = (mepll & 0xff00) >> 8  # bit 15:8
        me_clkoutdiv = math.pow(2, ((mepll & 0x30000) >> 16))

        me_core = int(os.popen(f"devmem {ME_CORE_REF_CTRL} 32").read(), 16)
        me_core_cal = (me_core & 0x3ff00) >> 8  # bit17:8
        mePLL_freq = hsm0_clk * mepll_cal / me_clkoutdiv / me_core_cal
        return mePLL_freq

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
            elif key == 'depth':
                depth = int(runtimeEventList.get(key, 0))
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
        self.vartThreads[pid]["depth"] = depth
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

    def timeline_dpucore_binding(self):
        for core in self.dpu_ip.cores:
            self.dpu_timelines[core.core_id].dpuCore = core

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
            event.depth = self.getDpuRuntimeInfo(event, "depth")

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

    def dpu_ip_add_query_info(self, data):

        ip_info = select_trace(data, "type", "DPU")[0].get("info")

        self.dpu_ip.ip_version = ip_info.get(
            "DPU IP Spec", {}).get("IP version", "")
        self.dpu_ip.timestamp = ip_info.get(
            "DPU IP Spec", {}).get("generation timestamp", "")
        self.dpu_ip.commit_id = ip_info.get(
            "DPU IP Spec", {}).get("git commit id", "")

        for cu_query in ip_info.get("kernels", []):
            for cu in self.dpu_ip.cores:
                if cu_query.get("is_vivado_flow", False) != True:
                    if cu_query.get("cu_name", "") != cu.full_name:
                        continue

                # matched!
                cu_arch_txt = cu_query.get("DPU Arch").split("_")
                cu.arch, cu.isa, cu.type = cu_arch_txt[0], cu_arch_txt[1], cu_arch_txt[2]

                cu.load_parallel = int(cu_query.get("Load Parallel", 0))
                cu.save_parallel = int(cu_query.get("Save Parallel", 0))

                cu.axilite_freq = int(cu_query.get(
                    "XRT Frequency (MHz)", 0)) * 1000 * 1000
                cu.pl_freq = int(cu_query.get(
                    "DPU Frequency (MHz)", 0)) * 1000 * 1000

                if (cu.arch == "DPUCVDX8G") or (cu.arch == "DPUCV2DX8G"):
                    cu.aie_freq = self.getDpuAieClk(cu.arch)
                    if (cu.aie_freq == 1):
                        cu.peak_ops = 1
                        return

                if cu.arch == "DPUCZDX8G":  # mpsoc
                    macs = int(cu.type[1:])
                    ops = (macs * (cu.pl_freq))
                elif (cu.arch == "DPUCVDX8G") or (cu.arch == "DPUCV2DX8G"):  # versal-1
                    cpb_list = re.findall(r'C(\d+)', cu.type)
                    cpb_n = int(cpb_list[0])
                    batch_list = re.findall(r'B(\d+)', cu.type)
                    batch_n = int(batch_list[0])
                    cu_n_judge = cu.type.find("CU")
                    if cu_n_judge != -1:
                        reg_cu_n = re.compile(r'(?<=CU)\d+')
                        cu_list = reg_cu_n.findall(cu.type)
                        cu_n = int(cu_list[0])
                    else:
                        cu_n = 1

                    if cu.arch == "DPUCVDX8G":
                        base = 256
                    if cu.arch == "DPUCV2DX8G":
                        base = 512

                    ops = base * cpb_n * batch_n * cu_n * cu.aie_freq
                else:
                    raise RuntimeError("unkwon version dpu")

                cu.peak_ops = ops

        # print(self.dpu_ip)

    def parse(self, data, dpu_ip_info, options):
        fg_mode = (options.get('runmode') == 'debug')

        """Two types of event tracing data included: dpuRuntimeEvent & dpuControllerEvent"""
        cuRetData = {}

        """get xmodel info from: {'classname': 'subgraph_info', 'info_file': '/tmp/vaitrace_subgraph_info_15230', 'section': 'INFO'}"""
        subgraph_info = select_trace(data, 'classname', "subgraph_info")
        self.xmodelInfo = xmodel_parser.xmodel_get_info(subgraph_info)

        # Check [RELEASE::SPLIT] mode for .xmodel
        if fg_mode:
            level2_subg = 0
            empty_level2_subg = 0
            for subg in self.xmodelInfo.values():
                if subg.get('depth', 0) == 2:
                    level2_subg += 1
                if subg.get('mc_size', 0) == 0:
                    empty_level2_subg += 1
            if (level2_subg == empty_level2_subg):
                logging.error("This xmodel does not support fine grained profiling")

        """Prepare at most 8 DPU timelines"""
        self.dpu_timelines = createTimelines('DPU', 8, options)
        time_offset = options.get("time_offset", 0)
        time_limit = options.get("time_limit", float('inf'))

        data = select_trace_classes(data, ["dpu-runner", "dpu-controller"])
        self.create_dpu_ip(select_trace(data, 'section', "INFO"))
        self.timeline_dpucore_binding()
        self.dpu_ip_add_query_info(dpu_ip_info)

        self.parse_trace(select_trace(data, 'section', "TRACE"))

        return self.dpu_timelines

    def get_dpu_ip_summary(self):
        return self.dpu_ip.summary()

    def get_dpu_profile_summary(self, options, fmt="vtf"):
        """
        DPU Profile Summary Format:
        Kernel Name,Number Of Runs,CU Full Name,Minimum Time (ms),Average Time (ms),Maximum Time (ms),Workload(GOP),DPU Performance(GOP/s),Mem IO(MB),Mem Bandwidth(MB/s),
        """

        runmachine = options.get('run_machine', {})
        subgraph_stat_time = {}
        subgraph_stat_hwcnt = {}
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
                if runmachine == "aarch64":
                    hwcounter = run_event.hwcounter

                if (subg_idx in self.xmodelInfo):
                    pass

                subgraph_stat_time.setdefault(dpu_id, {}).setdefault(
                    subg_idx, []).append(time)

                if runmachine == "aarch64":
                    subgraph_stat_hwcnt.setdefault(dpu_id, {}).setdefault(
                            subg_idx, []).append(hwcounter)

        for core_id in subgraph_stat_time.keys():
            for subg_idx in subgraph_stat_time[core_id]:
                subg_name = subg_idx.split('|')[0]
                times = sorted(subgraph_stat_time[core_id][subg_idx])

                min_t = times[0] * 1000
                max_t = times[-1] * 1000
                avg_t = sum(times) / len(times) * 1000
                runs = len(times)

                dpu_core = self.dpu_ip.cores[core_id]
                cu_name = dpu_core.name
                batch = dpu_core.batch
                ctrl_freq = dpu_core.axilite_freq

                if runmachine == "aarch64":
                    hw_cnt_stat = subgraph_stat_hwcnt[core_id][subg_idx]
                    hw_counter = sum(hw_cnt_stat) / len(hw_cnt_stat)
                    hw_rt = hw_counter / ctrl_freq * 1000
                elif runmachine == "x86_64":
                    hw_rt = avg_t

                display_name = "%s:batch-%d" % (cu_name, batch)

                workload = self.xmodelInfo.get(subg_idx, {}).get("workload", 0)
                depth = self.xmodelInfo.get(subg_idx, {}).get("depth", 0)
                it = str(self.xmodelInfo.get(subg_idx, {}).get(
                    "i_tensor_shape", "")).replace(",", "_")
                ot = str(self.xmodelInfo.get(subg_idx, {}).get(
                    "o_tensor_shape", "")).replace(",", "_")

                if runmachine == "aarch64":
                    if (dpu_core.peak_ops == 1):
                        logging.warning("Effic is runtime ops!")
                    effic = (workload * batch / hw_rt) / (dpu_core.peak_ops) * 1000
                elif runmachine == "x86_64":
                    effic = 0

                performance = workload * batch * 1000 / hw_rt #hw_rt = ms

                load_img_size = int(self.xmodelInfo.get(
                    subg_idx, {}).get("load_io_img_size", 0))
                load_para_size = int(self.xmodelInfo.get(
                    subg_idx, {}).get("load_io_para_size", 0))
                rank_id = int(self.xmodelInfo.get(
                    subg_idx, {}).get("rank_id", 0))

                write_fm_size = batch * int(self.xmodelInfo.get(
                    subg_idx, {}).get("save_io_size", 0))

                read_fm_size = load_img_size * batch

                read_wb_size = load_para_size

                mem_io = read_fm_size + read_wb_size + write_fm_size
                mem_io_bw = mem_io / hw_rt * 1000  # to MB
                #print(subg_name, cu_name, min_t, avg_t, max_t, runs, workload, write_fm_size)

                inst_size = int(self.xmodelInfo.get(subg_idx, {}).get("mc_size", 0))

                prt_data_set = {
                    "subg_name": subg_name,
                    "runs": runs,
                    "display_name": display_name,
                    "depth": depth,
                    "min_t": uConv(min_t),
                    "avg_t": uConv(avg_t),
                    "max_t": uConv(max_t),
                    "workload": uConv(workload, "Gi"),
                    "effic": uConv(effic, "%", 1),
                    "read_wb_size": uConv(read_wb_size, "MB"),
                    "read_fm_size": uConv(read_fm_size, "MB"),
                    "write_fm_size": uConv(write_fm_size, "MB"),
                    "mem_io_bw": uConv(mem_io_bw, "MB"),
                    "mem_io": uConv(mem_io, "MB"),
                    "hw_rt": uConv(hw_rt),
                    "rank_id": rank_id,
                    "inst_size": inst_size
                }

                txt_fmt = "{subg_name},{runs},{display_name},{min_t},{avg_t},{max_t},{workload},{effic},{read_wb_size},{read_fm_size},{write_fm_size},{mem_io_bw},{hw_rt},{rank_id},\n"
                vtf_fmt = "{subg_name},{runs},{display_name},{min_t:.3f},{avg_t:.3f},{max_t:.3f},{workload:.3f},{effic:.3f},{mem_io:.3f},{mem_io_bw:.3f},\n"
                json_fmt = f"{subg_name},{runs},{cu_name},{batch},{min_t},{avg_t},{max_t},{workload},{effic},{read_wb_size},{read_fm_size},{write_fm_size},{mem_io_bw},{hw_rt},{depth},{it},{ot},{performance},{inst_size}\n"

                if (fmt == "txt"):
                    ret.append(txt_fmt.format(**prt_data_set))
                elif (fmt == "json"):
                    ret.append(json_fmt)
                else:
                    ret.append(vtf_fmt.format(**prt_data_set))

        return ret

    def get_dpu_ips(self):
        return ips_profile(self.dpu_timelines)
