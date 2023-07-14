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

from typing import List
import numpy as np
import xir
import vart
import tools_extra_ops as tools
import os


def fillin_inputs(file_path_list: List["string"], tensor: "ndarray"):
    input_shape = tuple(tensor.shape[1:])
    for i in range(tensor.shape[0]):
        tensor[i, ...] = np.fromfile(file_path_list[i % len(file_path_list)],
                                     dtype=tensor.dtype,
                                     count=np.prod(input_shape)).reshape(input_shape)


def dump_outputs(outputs: List["ndarray"], tensors: List["Tensor"]):
    for output, tensor in zip(outputs, tensors):
        tensor_name = tools.remove_xfix(tensor.name).replace("/", "_")
        for batch_idx in range(output.shape[0]):
            output_file = str(batch_idx) + "." + tensor_name + ".bin"
            print("dump output to ", output_file)
            output[batch_idx, ...].tofile(output_file)


def run(subgraph: "Subgraph", args):
    # create runner
    runner = vart.Runner.create_runner(subgraph, args.runner_mode)

    # get input&output tensors
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    inputs = [
        np.empty(tuple(t.dims), dtype=t.dtype.lstrip("x"), order="C")
        for t in inputTensors
    ]
    outputs = [
        np.empty(tuple(t.dims), dtype=t.dtype.lstrip("x"), order="C")
        for t in outputTensors
    ]

    order_inputs = list(inputTensors)
    order = sorted(range(len(order_inputs)), key=lambda k: order_inputs[k].name)

    # fillin input data
    for i in range(len(inputTensors)):
        print("fillin", inputTensors[order[i]].name)
        fillin_inputs(args.input_bin[i::len(inputTensors)], inputs[order[i]])

    # run runner
    v = runner.execute_async(inputs, outputs)
    status = runner.wait(v)
    assert status == 0, "failed to run runner"

    dump_outputs(outputs, outputTensors)


def main(args):
    # get subgraph
    graph = xir.Graph.deserialize(args.xmodel)
    assert graph is not None, "'graph' should not be None."
    subgraph = graph.get_root_subgraph()
    assert (subgraph is not None), "Failed to get root subgraph of input Graph object."

    if args.subgraph_index >= 0:
        child_subgraphs = subgraph.toposort_child_subgraph()
        assert (len(child_subgraphs) > args.subgraph_index
                ), "cannot get child_subgraph[" + str(args.subgraph_index) + "]"
        subgraph = child_subgraphs[args.subgraph_index]
        disable_vart_cpu_runner = os.getenv("USE_VART_CPU_RUNNER", "null") == "null"
        if subgraph.has_attr("device") and subgraph.get_attr(
                "device").upper() == "CPU" and disable_vart_cpu_runner:
            subgraph.set_attr("runner", {"run": "libvitis_ai_library-cpu_task.so.3"})
    else:
        if not subgraph.has_attr("device"):
            subgraph.set_attr("device", "graph")
        if not subgraph.has_attr("runner"):
            subgraph.set_attr("runner", {"run": "libvitis_ai_library-graph_runner.so.3"})
    run(subgraph, args)


def help(subparsers):
    parser = subparsers.add_parser(
        "run",
        help=
        "<xmodel> [-i <subgraph_index>] [-r <run or ref or sim>] <input_tensor_0_bin_0> "
        + "[input_tensor_1_bin_0 input_tensor_0_bin_1 input_tensor_1_bin_1 ... ]",
    )
    parser.add_argument("xmodel", help="xmodel file path ")
    parser.add_argument("-i",
                        "--subgraph_index",
                        type=int,
                        default=1,
                        help="<subgraph_index>")
    parser.add_argument("-r",
                        "--runner_mode",
                        type=str,
                        default="run",
                        help="<run or ref or sim>")
    parser.add_argument("input_bin", nargs="+", help="input_bin")
    parser.set_defaults(func=main)
