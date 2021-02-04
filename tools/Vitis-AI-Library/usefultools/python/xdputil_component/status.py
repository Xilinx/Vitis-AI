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


import xir
import argparse
import json
import os
from typing import List

import tools_extra_ops as tools


def data_slice(source: "int", begin: "int", end: "int"):
    return (source >> begin) & (2 ** (end - begin) - 1)


def data_slice_2(source_h: "int", source_l: "int", begin: "int", end: "int"):
    return (((source_h << 32) + source_l) >> begin) & (2 ** (end - begin) - 1)


def status():
    addrs = {}
    infos = tools.device_info()
    dpu_arch = ""
    for info in infos:
        if info["fingerprint"]:
            dpu_arch = tools.get_target_type(info["fingerprint"])
            break
    res = {}
    if dpu_arch == "DPUCVDX8G":
        """xvdpu"""
        addrs = {
            "AP_REG": 0x00,
            "LSTART": 0x180,
            "LEND": 0x184,
            "CSTART": 0x188,
            "CEND": 0x18C,
            "SSTART": 0x190,
            "SEND": 0x194,
            "MSTART": 0x198,
            "MEND": 0x19C,
            "HP_BUS": 0x48,
            "INSTR_ADDR_L": 0x50,
            "INSTR_ADDR_H": 0x54,
        }
        for i in range(6):
            for j in range(4):
                addrs["Batch" + str(i) + "_addr" + str(j) + "_l"] = (
                    0x200 + (i * 4 + j) * 8
                )
                addrs["Batch" + str(i) + "_addr" + str(j) + "_h"] = (
                    0x204 + (i * 4 + j) * 8
                )

        res["DPU Registers"] = {}
        ap_status = {0b0001: "start", 0b0010: "done", 0b0100: "idle", 0b1000: "ready"}
        ap_reset_status = {0b01: "soft reset start", 0b10: "soft reset done"}
        for info in infos:
            dpu_idx = info["cu_idx"]
            read_res = tools.read_register("", dpu_idx, list(addrs.values()))
            if not len(read_res):
                continue
            source = dict(zip(list(addrs.keys()), read_res))

            register = {}
            """AP_REG"""
            register["Status"] = (
                ap_status[data_slice(source["AP_REG"], 0, 4)] + ""
                if not data_slice(source["AP_REG"], 5, 7)
                else "," + ap_reset_status[data_slice(source["AP_REG"], 5, 7)]
            )
            register["LOAD START"] = data_slice(source["LSTART"], 0, 32)
            register["LOAD END"] = data_slice(source["LEND"], 0, 32)
            register["SAVE START"] = data_slice(source["SSTART"], 0, 32)
            register["SAVE END"] = data_slice(source["SEND"], 0, 32)
            register["CONV START"] = data_slice(source["CSTART"], 0, 32)
            register["CONV END"] = data_slice(source["CEND"], 0, 32)
            register["MISC START"] = data_slice(source["MSTART"], 0, 32)
            register["MISC END"] = data_slice(source["MEND"], 0, 32)

            """"HP"""
            register["HP_AWCOUNT_MAX"] = data_slice(source["HP_BUS"], 24, 32)
            register["HP_ARCOUNT_MAX"] = data_slice(source["HP_BUS"], 16, 24)
            register["HP_AWLEN"] = data_slice(source["HP_BUS"], 8, 16)
            register["HP_ARLEN"] = data_slice(source["HP_BUS"], 0, 8)

            """"DPU code addr"""
            register["ADDR_CODE"] = hex(
                data_slice_2(source["INSTR_ADDR_H"], source["INSTR_ADDR_L"], 0, 64)
            )

            for i in range(6):
                for j in range(4):
                    register["Batch" + str(i) + "_addr" + str(j)] = hex(
                        data_slice_2(
                            source["Batch" + str(i) + "_addr" + str(j) + "_h"],
                            source["Batch" + str(i) + "_addr" + str(j) + "_l"],
                            0,
                            64,
                        )
                    )

            res["DPU Registers"]["DPU Core " + str(dpu_idx)] = register
    elif dpu_arch == "DPUCZDX8G":
        """zcu102/zcu104"""
        addrs = {
            "AP_REG": 0x00,
            "LSTART": 0x180,
            "LEND": 0x184,
            "CSTART": 0x188,
            "CEND": 0x18C,
            "SSTART": 0x190,
            "SEND": 0x194,
            "MSTART": 0x198,
            "MEND": 0x19C,
            "HP_BUS": 0x48,
            "INSTR_ADDR_L": 0x50,
            "INSTR_ADDR_H": 0x54,
        }
        for j in range(8):
            addrs["Base_addr" + str(j) + "_l"] = 0x60 + j * 8
            addrs["Base_addr" + str(j) + "_h"] = 0x64 + j * 8
        res["DPU Registers"] = {}

        ap_status = {0b0001: "start", 0b0010: "done", 0b0100: "idle", 0b1000: "ready"}
        ap_reset_status = {0b01: "soft reset start", 0b10: "soft reset done"}
        for info in infos:
            dpu_idx = info["cu_idx"]
            read_res = tools.read_register("", dpu_idx, list(addrs.values()))
            if not len(read_res):
                continue
            source = dict(zip(list(addrs.keys()), read_res))

            register = {}
            """AP_REG"""
            register["Status"] = (
                ap_status[data_slice(source["AP_REG"], 0, 4)] + ""
                if not data_slice(source["AP_REG"], 5, 7)
                else "," + ap_reset_status[data_slice(source["AP_REG"], 5, 7)]
            )
            register["LOAD START"] = data_slice(source["LSTART"], 0, 32)
            register["LOAD END"] = data_slice(source["LEND"], 0, 32)
            register["SAVE START"] = data_slice(source["SSTART"], 0, 32)
            register["SAVE END"] = data_slice(source["SEND"], 0, 32)
            register["CONV START"] = data_slice(source["CSTART"], 0, 32)
            register["CONV END"] = data_slice(source["CEND"], 0, 32)
            register["MISC START"] = data_slice(source["MSTART"], 0, 32)
            register["MISC END"] = data_slice(source["MEND"], 0, 32)

            """"HP"""
            register["HP_AWCOUNT_MAX"] = data_slice(source["HP_BUS"], 24, 32)
            register["HP_ARCOUNT_MAX"] = data_slice(source["HP_BUS"], 16, 24)
            register["HP_AWLEN"] = data_slice(source["HP_BUS"], 8, 16)
            register["HP_ARLEN"] = data_slice(source["HP_BUS"], 0, 8)

            """"DPU code addr"""
            register["ADDR_CODE"] = hex(
                data_slice_2(source["INSTR_ADDR_H"], source["INSTR_ADDR_L"], 0, 64)
            )

            for j in range(8):
                register["Base_addr" + str(j)] = hex(
                    data_slice_2(
                        source["Base_addr" + str(j) + "_h"],
                        source["Base_addr" + str(j) + "_l"],
                        0,
                        64,
                    )
                )

            res["DPU Registers"]["DPU Core " + str(dpu_idx)] = register
    elif dpu_arch == "DPUCAHX8H":
        """cloud"""
        # print(json.dumps(infos, sort_keys=True, indent=4, separators=(",", ":")))
        addrs = {
            "AP_REG": 0x000,
            "LSTART": 0x0A0,
            "LEND": 0x090,
            "SSTART": 0x09C,
            "SEND": 0x08C,
            "CSTART": 0x098,
            "CEND": 0x088,
            "MSTART": 0x094,
            "MEND": 0x084,
            "reg_hp_setting": 0x020,
            "INSTR_ADDR_L": 0x140,
            "INSTR_ADDR_H": 0x144,
        }
        i = 0
        for j in range(8):
            addrs["dpu" + str(i) + "_reg_base_addr_" + str(j) + "_l"] = (
                0x100 + (i * 4 + j) * 8
            )
            addrs["dpu" + str(i) + "_reg_base_addr_" + str(j) + "_h"] = (
                0x104 + (i * 4 + j) * 8
            )
        for i in range(1, 8):
            for j in range(8):
                addrs["dpu" + str(i) + "_reg_base_addr_" + str(j) + "_l"] = (
                    0x200 + (i * 4 + j) * 8
                )
                addrs["dpu" + str(i) + "_reg_base_addr_" + str(j) + "_h"] = (
                    0x204 + (i * 4 + j) * 8
                )
        res = {}
        ap_status = {0b0001: "start", 0b0010: "done", 0b0100: "idle"}
        for info in infos:
            dpu_idx = info["cu_idx"]
            read_res = tools.read_register("", dpu_idx, list(addrs.values()))
            if not len(read_res):
                continue
            source = dict(zip(list(addrs.keys()), read_res))

            register = {}
            """AP_REG"""
            register["AP status"] = ap_status[data_slice(source["AP_REG"], 0, 3)]

            register["LOAD START"] = data_slice(source["LSTART"], 0, 32)
            register["LOAD END"] = data_slice(source["LEND"], 0, 32)
            register["SAVE START"] = data_slice(source["SSTART"], 0, 32)
            register["SAVE END"] = data_slice(source["SEND"], 0, 32)
            register["CONV START"] = data_slice(source["CSTART"], 0, 32)
            register["CONV END"] = data_slice(source["CEND"], 0, 32)
            register["MISC START"] = data_slice(source["MSTART"], 0, 32)
            register["MISC END"] = data_slice(source["MEND"], 0, 32)

            register["HP_COUNT_MAX"] = data_slice(source["reg_hp_setting"], 16, 24)
            register["HP_AWLEN"] = data_slice(source["reg_hp_setting"], 8, 16)
            register["HP_ARLEN"] = data_slice(source["reg_hp_setting"], 0, 8)

            """"DPU code addr"""
            register["ADDR_CODE"] = hex(
                data_slice_2(source["INSTR_ADDR_H"], source["INSTR_ADDR_L"], 0, 64)
            )

            for i in range(8):
                for j in range(8):
                    register["dpu" + str(i) + "_reg_base_addr_" + str(j)] = hex(
                        data_slice_2(
                            source["dpu" + str(i) + "_reg_base_addr_" + str(j) + "_h"],
                            source["dpu" + str(i) + "_reg_base_addr_" + str(j) + "_l"],
                            0,
                            64,
                        )
                    )
            res["cu_idx : " + str(dpu_idx)] = register
    else:
        res["Unsupported platform type"] = dpu_arch
    return res


def main(args):
    print(json.dumps(status(), sort_keys=False, indent=4, separators=(",", ":")))


def help(subparsers):
    parser = subparsers.add_parser("status", help="")

    parser.set_defaults(func=main)
