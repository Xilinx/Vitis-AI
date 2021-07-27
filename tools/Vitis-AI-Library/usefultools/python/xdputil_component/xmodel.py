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
import sys
from typing import List
import tools_extra_ops as tools
import hashlib
import json


def png(args):
    xir.Graph.deserialize(args.xmodel).save_as_image(args.png, "png")


def svg(args):
    xir.Graph.deserialize(args.xmodel).save_as_image(args.svg, "svg")


def txt(args):
    import tools_extra_ops as tools

    res = tools.xmodel_to_txt(args.xmodel)
    if args.txt != sys.stdout:
        open(args.txt, mode="w").write(res)
    else:
        print(res)


def get_op_output_tensor_ddr(op: "Op", s: "Subgraph"):
    tensor = op.get_output_tensor()
    if op.get_type() == "download" and op in s.get_ops():
        input_ops = op.get_input_ops_by_name("input")
        assert len(input_ops) == 1, "There must be only one pre_op for download op."
        tensor = input_ops[0].get_output_tensor()
    elif not tensor.has_attr("reg_id") or not op in s.get_ops():
        ops = list(set(op.get_fanout_ops()) & s.get_ops())
        assert len(ops) == 1, (
            "illegal xmodel. op:" + op.get_name() + "  has no ddr info"
        )
        assert ops[0].get_type() == "upload", (
            "illegal xmodel. op:" + op.get_name() + "  has no ddr info"
        )
        tensor = ops[0].get_output_tensor()
    assert tensor.has_attr("reg_id"), "op_name " + op.get_name()
    reg_id = tensor.get_attr("reg_id")

    return reg_id


def get_tensor_ddr_info(s: "Subgraph", tensors):
    return {get_op_output_tensor_ddr(t.producer, s) for t in tensors}


def create_dpu_result(s: "Subgraph"):
    result_s = {}
    result_s["fingerprint"] = (
        hex(0)
        if not s.has_attr("dpu_fingerprint")
        else hex(s.get_attr("dpu_fingerprint"))
    )
    result_s["DPU Arch"] = s.get_attr("dpu_name")

    result_s["input_tensor"] = [
        {
            "name": t.name,
            "shape": str(t.dims),
            "fixpos": t.get_attr("fix_point"),
        }
        for t in s.get_input_tensors()
    ]
    result_s["output_tensor"] = [
        {
            "name": t.name,
            "shape": str(t.dims),
            "fixpos": t.get_attr("fix_point"),
        }
        for t in s.get_output_tensors()
    ]
    # result_s["subgraph"] = [c.get_name() for c in s.toposort_child_subgraph()]
    """
    if s.has_attr("reg_id_to_context_type"):
        cont_type = s.get_attr("reg_id_to_context_type")
        table = {k: [v] for k, v in cont_type.items()}
    """
    input_id_set = get_tensor_ddr_info(s, s.get_input_tensors())
    output_id_set = get_tensor_ddr_info(s, s.get_output_tensors())
    cont_type = {}
    if s.has_attr("reg_id_to_context_type_v2"):
        cont_type_2 = s.get_attr("reg_id_to_context_type_v2")
        for k, v in cont_type_2.items():
            if v == "INTERFACE":
                if int(k[-1]) in input_id_set and int(k[-1]) in output_id_set:
                    v = "DATA_LOCAL"
                elif int(k[-1]) in input_id_set:
                    v = "DATA_LOCAL_INPUT"
                elif int(k[-1]) in output_id_set:
                    v = "DATA_LOCAL_OUTPUT"
                else:
                    v = "INVALID TYPE: " + v
            cont_type[k] = v
    elif s.has_attr("reg_id_to_context_type"):
        cont_type = s.get_attr("reg_id_to_context_type")

    pmd5 = tools.get_reg_id_to_parameter(s)

    size = {}
    if s.has_attr("reg_id_to_size"):
        size = s.get_attr("reg_id_to_size")
        for k, v in size.items():
            size[k] = str(round(v / 1024.0 / 1024.0, 2)) + "MB"

    keys = list(set(list(cont_type.keys()) + list(pmd5.keys()) + list(size.keys())))
    keys.sort()
    table = [
        {
            "name": k,
            "context type": cont_type.get(k),
            "parameter md5 value": pmd5.get(k),
            "memory size required": size.get(k),
        }
        for k in keys
    ]

    result_s["reg info"] = table
    return result_s


def get_child_subgraph(graph: "Graph") -> List["Subgraph"]:
    """
    obtain subgrah
    """
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    return root_subgraph.toposort_child_subgraph()


def glist(xmodel_path: "String"):
    graph = xir.Graph.deserialize(xmodel_path)
    res = {}
    res["subgraphs"] = []
    for cs in get_child_subgraph(graph):
        device = "" if not cs.has_attr("device") else cs.get_attr("device").upper()
        if device:
            result_s = {}
            result_s["name"] = cs.get_name()
            result_s["device"] = device
            if device == "DPU":
                result_s = dict(result_s, **create_dpu_result(cs))
            res["subgraphs"].append(result_s)

    print(json.dumps(res, sort_keys=False, indent=4, separators=(",", ":")))


def meta_info(xmodel_path: "String"):
    graph = xir.Graph.deserialize(xmodel_path).get_attrs()
    for k in graph["files_md5sum"].keys():
        graph["files_md5sum"][k.split("/")[-1]] = graph["files_md5sum"].pop(k)

    for k in graph["files_md5sum"].keys():
        if k.find("_org.xmodel") != -1:
            graph["files_md5sum"].pop(k)
            break

    print(json.dumps(graph, sort_keys=True, indent=4, separators=(",", ":")))


def xmodel_main(args):
    if args.png:
        png(args)
    elif args.svg:
        svg(args)
    elif args.txt:
        txt(args)
    elif args.list:
        glist(args.xmodel)
    elif args.meta_info:
        meta_info(args.xmodel)


def help(subparsers):
    parser = subparsers.add_parser(
        "xmodel",
        description="xmodel ",
        help="<xmodel> [-h] | [-l] | [-m] | [-p png] | [-s svg] | [-t txt]",
    )
    parser.add_argument("xmodel", help="xmodel file path ")
    parser.add_argument(
        "-l",
        "--list",
        help="show subgraph list",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--meta_info",
        help="show xcompiler version",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--png",
        help="the output to png ",
        nargs="?",
        const="default.png",
    )
    parser.add_argument(
        "-s",
        "--svg",
        help="the output svg path ",
        nargs="?",
        const="default.svg",
    )
    parser.add_argument(
        "-t",
        "--txt",
        help="when <txt> is missing, it dumps to standard output.",
        nargs="?",
        const=sys.stdout,
    )

    parser.set_defaults(func=xmodel_main)
