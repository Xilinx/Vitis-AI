#!/usr/bin/env python
# coding=utf-8
"""
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import json

import tools_extra_ops as tools
from .status import data_slice


def create_info(info):
    if info["fingerprint"]:
        info["DPU Arch"] = tools.get_target_name(info["fingerprint"])

    info["fingerprint"] = hex(info["fingerprint"])
    info["cu_handle"] = hex(info["cu_handle"])
    info["cu_addr"] = hex(info["cu_addr"])
    return info


def create_vai_version():
    so_list = [
        "libxir.so",
        "libvart-dpu-runner.so",
        "libvart-mem-manager.so",
        "libvart-runner.so",
        "libvart-dpu-controller.so",
        "libvart-softmax-runner.so",
        "libvart-xrt-device-handle.so",
        "libvitis_ai_library-dpu_task.so",
    ]
    res = dict(zip(so_list, tools.xilinx_version(so_list)))
    res["target_factory"] = (
        tools.get_target_factory_name() + " " + tools.get_target_factory_id()
    )

    return res


def whoami():
    infos = tools.device_info()
    res = {}
    dpu_arch = ""
    for info in infos:
        if info["fingerprint"]:
            dpu_arch = tools.get_target_type(info["fingerprint"])
            break
    if dpu_arch == "DPUCVDX8G" or dpu_arch == "DPUCZDX8G":
        addrs = {
            "SYS": 0x20,
            "SUB_VERSION": 0x108,
            "TIMESTAMP": 0x24,
            "GIT_COMMIT_ID": 0x100,
            "GIT_COMMIT_TIME": 0x104,
        }
        ip_type = {
            1: "DPU",
            2: "softmax",
            3: "sigmoid",
            4: "resize",
            5: "SMFC",
            6: "YRR",
        }

        read_res = tools.read_register("DPU", 0, list(addrs.values()))
        source = dict(zip(list(addrs.keys()), read_res))
        # print(json.dumps(source, sort_keys=True, indent=4, separators=(",", ":")))
        dpu_inf = {}
        """GIT_COMMIT_ID"""
        dpu_inf["git commit id"] = format(
            data_slice(source["GIT_COMMIT_ID"], 0, 28), "x"
        )
        """GIT_COMMIT_TIME"""
        dpu_inf["git commit time"] = data_slice(source["GIT_COMMIT_TIME"], 0, 32)

        """TIMESTAMP yyyy-MM-dd HH-mm-ss """
        dpu_inf["generation timestamp"] = (
            "20"
            + str(data_slice(source["TIMESTAMP"], 24, 32))
            + "-"
            + str(data_slice(source["TIMESTAMP"], 20, 24)).zfill(2)
            + "-"
            + str(data_slice(source["TIMESTAMP"], 12, 20)).zfill(2)
            + " "
            + str(data_slice(source["TIMESTAMP"], 4, 12)).zfill(2)
            + "-"
            + str(data_slice(source["TIMESTAMP"], 0, 4) * 15).zfill(2)
            + "-00"
        )

        """SYS"""
        dpu_inf["IP version"] = (
            "v"
            + ".".join(list(format(data_slice(source["SYS"], 24, 32), "x")))
            + "."
            + format(data_slice(source["SUB_VERSION"], 12, 20), "x")
        )
        dpu_inf["DPU Target Version"] = "v" + ".".join(
            format(data_slice(source["SUB_VERSION"], 0, 12), "x")
        )

        regmap_version = {
            0: "Initial version.",
            1: "1toN version",
            2: "1to1 version",
        }
        dpu_inf["regmap"] = regmap_version[data_slice(source["SYS"], 0, 8)]
        dpu_inf["DPU Core Count"] = len(infos)
        dpu_inf["DPU Core Count"] = len(infos)
        res["DPU IP Spec"] = dpu_inf

        if dpu_arch == "DPUCVDX8G":
            addrs = {"BATCH_N": 0x134, "SYS": 0x20, "FREQ": 0x28}

        if dpu_arch == "DPUCZDX8G":
            """zcu102/zcu104"""
            addrs = {"SYS": 0x20, "FREQ": 0x28}

        for info in infos:
            dpu_idx = info["cu_idx"]
            read_res = tools.read_register("", dpu_idx, list(addrs.values()))
            if not len(read_res):
                continue
            source = dict(zip(list(addrs.keys()), read_res))
            dpu = create_info(info)
            if dpu_arch == "DPUCVDX8G":
                dpu["DPU Batch Number"] = data_slice(source["BATCH_N"], 0, 4)
            dpu["IP Type"] = ip_type[data_slice(source["SYS"], 16, 24)]
            dpu["DPU Frequency (MHz)"] = data_slice(source["FREQ"], 0, 12)
            res["DPU Core : # " + str(info["cu_idx"])] = dpu
    elif dpu_arch == "DPUCAHX8H":
        addrs = {"HARDWARE_VER_1": 0x1F0, "HARDWARE_VER_2": 0x1F4}

        read_res = tools.read_register("DPU", 0, list(addrs.values()))
        source = dict(zip(list(addrs.keys()), read_res))

        hard_ver = (source["HARDWARE_VER_2"] << 32) + source["HARDWARE_VER_1"]
        dpu_inf = {}
        dpu_inf["dwc"] = "enable" if data_slice(hard_ver, 0, 1) else "disable"
        dpu_inf["leakyrelu"] = "enable" if data_slice(hard_ver, 1, 2) else "disable"
        dpu_inf["misc_parallesim"] = "2p" if data_slice(hard_ver, 2, 3) else "1p"
        dpu_inf["Bank Group Volume"] = (
            "VB"
            if data_slice(hard_ver, 3, 5) == 0b11
            else (
                "2MB BKG"
                if data_slice(hard_ver, 3, 5) == 0b01
                else ("512KB BKG" if data_slice(hard_ver, 3, 5) == 0b00 else "none")
            )
        )
        dpu_inf["long weight"] = (
            "support" if data_slice(hard_ver, 5, 6) else "not support"
        )
        dpu_inf["pooling kernel size 5x5 operation"] = (
            "support" if data_slice(hard_ver, 6, 7) else "not support"
        )
        dpu_inf["pooling kernel size 8x8 operation"] = (
            "support" if data_slice(hard_ver, 7, 8) else "not support"
        )
        dpu_inf["pooling kernel size 4x4 operation"] = (
            "support" if data_slice(hard_ver, 8, 9) else "not support"
        )
        dpu_inf["pooling kernel size 6x6 operation"] = (
            "support" if data_slice(hard_ver, 9, 10) else "not support"
        )
        dpu_inf["isa encoding"] = data_slice(hard_ver, 48, 56)
        dpu_inf["ip encoding"] = data_slice(hard_ver, 56, 64)
        res["DPU IP Spec"] = dpu_inf

        for info in infos:
            dpu = create_info(info)
            res["DPU Core : # " + str(info["cu_idx"])] = dpu

    return res


def main(args):
    res = whoami()
    res["VAI Version"] = create_vai_version()
    print(json.dumps(res, sort_keys=True, indent=4, separators=(",", ":")))


def help(subparsers):
    parser = subparsers.add_parser("query", help="No input parameters")
    parser.set_defaults(func=main)
