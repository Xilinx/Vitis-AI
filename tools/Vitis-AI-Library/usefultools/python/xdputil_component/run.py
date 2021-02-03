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


from typing import List
import numpy as np
import xir
import vart
import tools_extra_ops as tools


def fillin_inputs(file_path_list: List["string"], tb: "TensorBuffer"):
    tensor = np.array(tb, copy=False)
    input_shape = tuple(tensor.shape[1:])
    for i in range(tensor.shape[0]):
        tensor[i, ...] = np.fromfile(
            file_path_list[i % len(file_path_list)],
            dtype=np.uint8,
            count=np.prod(input_shape),
        ).reshape(input_shape)


def dump_outputs(tbs: List["TensorBuffer"]):
    for tb in tbs:
        tensor_name = tools.remove_xfix(tb.get_tensor().name).replace("/", "_")
        tensor = np.array(tb, copy=False)
        # print("batch is ", tensor.shape[0])
        for batch_idx in range(tensor.shape[0]):
            output_file = str(batch_idx) + "." + tensor_name + ".bin"
            print("dump output to ", output_file)
            tensor[batch_idx, ...].tofile(output_file)


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    """
    obtain dpu subgrah
    """
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def main(args):
    # get subgraph
    graph = xir.Graph.deserialize(args.xmodel)
    child_subgraph = get_child_subgraph_dpu(graph)
    assert len(child_subgraph) > args.subgraph_index, (
        "cannot get child_subgraph[" + str(args.subgraph_index) + "]"
    )

    # create runner
    runner = vart.RunnerExt.create_runner(child_subgraph[args.subgraph_index], "run")

    """get input&output  tensor_buffers"""
    inputs = runner.get_inputs()
    outputs = runner.get_outputs()

    """ fillin input data """
    fillin_inputs(args.input_bin, inputs[0])

    # run dpu
    v = runner.execute_async(inputs, outputs)
    status = runner.wait(v)
    assert status == 0, "failed to run dpu"

    dump_outputs(outputs)


def help(subparsers):
    parser = subparsers.add_parser(
        "run", help="<xmodel> [-i <subgraph_index>] <input_bin>"
    )
    parser.add_argument("xmodel", help="xmodel file path ")
    parser.add_argument(
        "-i", "--subgraph_index", type=int, default=0, help="<subgraph_index>"
    )
    parser.add_argument("input_bin", nargs="+", help="input_bin ")
    parser.set_defaults(func=main)
