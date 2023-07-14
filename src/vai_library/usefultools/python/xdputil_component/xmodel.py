#!/usr/bin/env python
# coding=utf-8
"""
Copyright 2022-2023 Advanced Micro Devices Inc.

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
import os
import sys
import os
from typing import List
import tools_extra_ops as tools
import hashlib
import json
import numpy as np


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
        assert len(ops) == 1, ("illegal xmodel. op:" + op.get_name() +
                               "  has no ddr info")
        assert ops[0].get_type() == "upload", ("illegal xmodel. op:" + op.get_name() +
                                               "  has no ddr info")
        tensor = ops[0].get_output_tensor()
    assert tensor.has_attr("reg_id"), "op_name " + op.get_name()
    reg_id = tensor.get_attr("reg_id")

    return reg_id


def get_tensor_ddr_info(s: "Subgraph", tensors):
    return {get_op_output_tensor_ddr(t.producer, s) for t in tensors}


def create_dpu_result(s: "Subgraph"):
    result_s = {}
    result_s["fingerprint"] = (hex(0) if not s.has_attr("dpu_fingerprint") else hex(
        s.get_attr("dpu_fingerprint")))
    result_s["DPU Arch"] = s.get_attr("dpu_name")
    result_s["workload"] = s.get_attr("workload")

    input_tensors = list(s.get_input_tensors())
    input_tensors.sort(key=lambda tensor: tensor.name)

    result_s["input_tensor"] = [{
        "index": idx,
        "name": t.name,
        "shape": t.dims,
        "fixpos": t.get_attr("fix_point"),
    } for idx, t in enumerate(input_tensors)]
    result_s["output_tensor"] = [{
        "index": idx,
        "name": t.name,
        "shape": t.dims,
        "fixpos": t.get_attr("fix_point"),
    } for idx, t in enumerate(s.get_output_tensors())]
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
                reg_id = int(k.replace("REG_",""))
                if reg_id in input_id_set and reg_id in output_id_set:
                    v = "DATA_LOCAL"
                elif reg_id in input_id_set:
                    v = "DATA_LOCAL_INPUT"
                elif reg_id in output_id_set:
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
        # for k, v in size.items():
        #     size[k] = str(round(v / 1024.0 / 1024.0, 2)) + "MB"

    keys = list(set(list(cont_type.keys()) + list(pmd5.keys()) + list(size.keys())))
    keys.sort()
    table = [
        {
            "name": k,
            "context type": cont_type.get(k),
            # "parameter md5 value": pmd5.get(k),
            "size": size.get(k),
        } for k in keys
    ]

    result_s["reg info"] = table

    if s.has_attr("mc_code"):
        ins_reg = s.get_attr("mc_code")
        result_s["instruction reg"] = len(ins_reg)

    return result_s


def get_child_subgraph(graph: "Graph") -> List["Subgraph"]:
    """
    obtain subgrah
    """
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    return root_subgraph.toposort_child_subgraph()


def glist(xmodel_path: "String"):
    graph = xir.Graph.deserialize(xmodel_path)
    res = {}
    res["subgraphs"] = []
    idx = 0
    for cs in get_child_subgraph(graph):
        device = "" if not cs.has_attr("device") else cs.get_attr("device").upper()
        if device:
            result_s = {}
            result_s["index"] = idx
            idx += 1
            result_s["name"] = cs.get_name()
            result_s["device"] = device
            if device == "DPU":
                result_s = dict(result_s, **create_dpu_result(cs))
            res["subgraphs"].append(result_s)

    print(json.dumps(res, sort_keys=False, indent=4, separators=(",", ":")))


input_args_type = ("input", "weights", "bias")
# attrs_filter_out = ("data", "quant_out_signed", "quant_out_round_mode", "quant_in_signed")


def op_info(xmodel: "String", op_name: "String"):
    print("xmodel: ", xmodel)
    print("op_name: ", op_name)
    graph = xir.Graph.deserialize(xmodel)
    op = graph.get_op(op_name)
    if op is None:
        print("Error: op name doesn't exist in the model!")
        return

    res = {}
    res["name"] = op.get_name()
    res["type"] = op.get_type()
    op_attrs = op.get_attrs()
    filter_attrs = {}
    for k, v in op_attrs.items():
        if k != "data" and not k.startswith("quant_"):
            filter_attrs[k] = v

    # for attr in attrs_filter_out:
    #     if op_attrs.get(attr) is not None:
    #         op_attrs.pop(attr)

    res["attrs"] = filter_attrs
    input_ops = op.get_input_ops()
    op_list = []
    if len(input_ops):
        op_index = 0
        for arg in input_args_type:
            if input_ops.get(arg) is not None:
                args = input_ops[arg]
                for idx in args:
                    op_dict = {}
                    op_dict["index"] = op_index
                    op_index += 1
                    op_dict["op_name"] = idx.get_name()
                    op_dict["tensor_name"] = idx.get_output_tensor().name
                    op_dict["shape"] = idx.get_output_tensor().dims
                    op_dict["data_type"] = idx.get_output_tensor().dtype
                    op_list.append(op_dict)
    res["inputs"] = op_list

    out_res = {}
    out_res["op_name"] = op.get_name()
    out_res["tensor_name"] = op.get_output_tensor().name
    out_res["shape"] = op.get_output_tensor().dims
    out_res["data_type"] = op.get_output_tensor().dtype
    out_res_attrs = {}
    for k, v in op.get_output_tensor().get_attrs().items():
        if not k.startswith("quant_"):
            out_res_attrs[k] = v
    out_res["attrs"] = out_res_attrs
    res["outputs"] = out_res

    print(json.dumps(res, sort_keys=False, indent=4, separators=(",", " : ")))


def meta_info(xmodel_path):
    graph = xir.Graph.deserialize(xmodel_path).get_attrs()
    for k in list(graph["files_md5sum"].keys()):
        graph["files_md5sum"][k.split("/")[-1]] = graph["files_md5sum"].pop(k)

    for k in graph["files_md5sum"].keys():
        if k.find("_org.xmodel") != -1:
            graph["files_md5sum"].pop(k)
            break

    print(json.dumps(graph, sort_keys=True, indent=4, separators=(",", ":")))


def node_id(i):
    return 'node{0}'.format(i)


def get_attr(tensor, attr_name, default_value):
    if not tensor.has_attr(attr_name):
        return default_value
    return tensor.get_attr(attr_name)


def node_label(i, sg):
    return '''{0}: {1}'''.format(i, sg.get_name())


def find_subgraph_id(g, tensor, sgs):
    op = g.get_tensor_producer(tensor)
    s = g.get_leaf_subgraph(op)
    index = -1
    while not s == g.get_root_subgraph():
        try:
            index = sgs.index(s)
            break
        except ValueError:
            s = s.get_parent()
    assert not index == -1
    return index


def tensor_label(tensor):
    return '''
name: {0}
type: {1}
fix_point: {2}
shape: {3}
'''.format(tensor.name, tensor.producer.get_type(), get_attr(tensor, 'fix_point', 'N/A'),
           tensor.dims)


def subgraph_svg(xmodel_path: "String", svg_file: "String"):
    graph = xir.Graph.deserialize(xmodel_path)
    from graphviz import Digraph
    g = Digraph()  # comment='The Round Table')
    g.attr(compound='true')
    sgs = graph.get_root_subgraph().toposort_child_subgraph()
    i = 0
    for sg in sgs:
        with g.subgraph(name='cluster%d' % (i, )) as c:
            c.attr('node', shape='plaintext')
            c.attr(label=sg.get_name())
            # node_label(i, sg))
            c.node(node_id(i), label=sg.get_attr('device'))
            output_tensors = list(sg.get_output_tensors())
            output_tensors.sort(key=lambda tensor: tensor.name)
            for tensor in output_tensors:
                c.attr('node', shape='box')
                c.node(tensor.name, tensor_label(tensor))
                c.edge(node_id(i), tensor.name)
        i = i + 1
    i = 0
    for sg in sgs:
        for tensor in sg.get_input_tensors():
            g.edge(tensor.name, node_id(i), lhead="cluster%d" % (i, ))
        i = i + 1
    filename, file_extension = os.path.splitext(svg_file)
    file_extension = file_extension[1:]  # remove "."
    g.render(filename=filename, cleanup=False, format=file_extension)


def binary(xmodel_path: "String", output_directory: "String"):
    graph = xir.Graph.deserialize(xmodel_path)
    children = graph.get_root_subgraph().toposort_child_subgraph()
    index = 0
    for sg in children:
        sg_path = os.path.join(output_directory, "sg_{}".format(index))
        os.makedirs(sg_path, exist_ok=True)
        if sg.has_attr('mc_code'):
            buf = tools.xir_get_attr_binary(sg, 'mc_code')
            array = np.array(buf, copy=False)
            filename = os.path.join(sg_path, "mc_code.bin")
            print("write to {}".format(filename))
            array.tofile(filename)
        if sg.has_attr('reg_id_to_parameter_value'):
            map_buf = tools.xir_get_attr_map_binary(sg, "reg_id_to_parameter_value")
            for (key, buf) in map_buf.items():
                array = np.array(buf, copy=False)
                filename = os.path.join(sg_path, "{}.bin".format(key))
                print("write to {}".format(filename))
                array.tofile(filename)


def xmodel_main(args):
    if args.png:
        png(args)
    elif args.svg:
        svg(args)
    elif args.txt:
        txt(args)
    elif args.list:
        glist(args.xmodel)
    elif args.op:
        op_info(args.xmodel, args.op)
    elif args.meta_info:
        meta_info(args.xmodel)
    elif args.subgraph_svg:
        subgraph_svg(args.xmodel, args.subgraph_svg)
    elif args.binary:
        binary(args.xmodel, args.binary)


def help(subparsers):
    parser = subparsers.add_parser(
        "xmodel",
        description="xmodel ",
        help=
        "<xmodel> [-h] | [-l] | [-m] | [-p png] | [-s svg] | [-t txt] | [--op op_name]",
    )
    parser.add_argument("xmodel", help="xmodel file path ")
    parser.add_argument(
        "-l",
        "--list",
        help="show subgraph list",
        action="store_true",
    )
    parser.add_argument(
        "--op",
        help="show op info",
        nargs="?",
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
        "-S",
        "--subgraph_svg",
        help="the output svg for subgraph level ",
        nargs="?",
        const="subgraph.svg",
    )
    parser.add_argument(
        "-t",
        "--txt",
        help="when <txt> is missing, it dumps to standard output.",
        nargs="?",
        const=sys.stdout,
    )
    parser.add_argument(
        "-b",
        "--binary",
        help=
        '''dump the binary data to the output directory, when <binary> is missing, it dumps to 'binary' directory''',
        nargs="?",
        const="binary",
    )

    parser.set_defaults(func=xmodel_main)
