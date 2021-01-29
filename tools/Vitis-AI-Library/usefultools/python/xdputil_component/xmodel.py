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
            "# of elements": hex(t.get_element_num()),
        }
        for t in s.get_input_tensors()
    ]
    result_s["output_tensor"] = [
        {
            "name": t.name,
            "shape": str(t.dims),
            "fixpos": t.get_attr("fix_point"),
            "# of elements": hex(t.get_element_num()),
        }
        for t in s.get_output_tensors()
    ]
    # result_s["subgraph"] = [c.get_name() for c in s.toposort_child_subgraph()]
    table = {}
    if s.has_attr("reg_id_to_context_type"):
        cont_type = s.get_attr("reg_id_to_context_type")
        table = {k: [v] for k, v in cont_type.items()}

    """
    if cs.has_attr("reg_id_to_context_type_v2"):
        cont_type = cs.get_attr("reg_id_to_context_type_v2")
        for k, v in cont_type.items():
            if k in table:
                table[k].append(v)
            else:
                table[k] = [v]
    else:
        print("not have _v2")
    """

    if s.has_attr("reg_id_to_parameter_value"):
        size = tools.get_reg_id_to_parameter(s)
        for k, v in size.items():
            if k in table:
                table[k].append(hashlib.md5(bytes(list(map(ord, v)))).hexdigest())
            else:
                table[k] = [hashlib.md5(bytes(list(map(ord, v)))).hexdigest()]

    if s.has_attr("reg_id_to_size"):
        size = s.get_attr("reg_id_to_size")
        for k, v in size.items():
            table[k].append(str(round(v / 1024.0 / 1024.0, 2)) + "MB")
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
    for cs in get_child_subgraph(graph):
        device = "" if not cs.has_attr("device") else cs.get_attr("device").upper()
        if device:
            result_s = {}
            result_s["device"] = device
            if device == "DPU":
                result_s = dict(result_s, **create_dpu_result(cs))
            res[cs.get_name()] = result_s

    print(json.dumps(res, sort_keys=False, indent=4, separators=(",", ":")))


def meta_info(xmodel_path: "String"):
    graph = xir.Graph.deserialize(xmodel_path).get_attrs()
    for k in graph["files_md5sum"].keys():
        graph["files_md5sum"][k.split("/")[-1]] = graph["files_md5sum"].pop(k)

    for k in graph["files_md5sum"].keys():
        if k.find("xmodel") != -1:
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
        help="<xmodel> [-h] | [-l] | [-i] | [-p png] | [-s svg] | [-t txt]",
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
