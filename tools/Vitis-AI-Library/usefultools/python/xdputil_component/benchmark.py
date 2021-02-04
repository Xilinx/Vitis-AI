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


import tools_extra_ops as tools
from .run import get_child_subgraph_dpu
import xir


def get_child_subgraph_dpu_i(xmodel: "String", subgraph_index: "int") -> "String":
    graph = xir.Graph.deserialize(xmodel)
    child_subgraph = get_child_subgraph_dpu(graph)
    assert len(child_subgraph) > subgraph_index, (
        "cannot get child_subgraph[" + str(subgraph_index) + "]"
    )
    return child_subgraph[subgraph_index].get_name()


def main(args):
    subgraph_name = get_child_subgraph_dpu_i(args.xmodel, args.subgraph_index)
    if tools.test_dpu_runner_mt(args.xmodel, subgraph_name, args.num_of_threads):
        print("Test PASS.")
    else:
        print("Test Failed!!!")


def help(subparsers):
    from argparse import RawTextHelpFormatter

    parser = subparsers.add_parser(
        "benchmark",
        help="<xmodel> [-i subgraph_index] <num_of_threads>",
        description=(
            "env variables:\n"
            + "\tCOPY_INPUT=1   : enable copying input\n"
            + "\tCOPY_OUTPUT=1  : enable copying output\n"
            + "\tENABLE_MEMCMP=1: enable comparing\n"
            + "\tSLEEP_MS=60000 : sleep for 60s before stopping\n"
            + "\tNUM_OF_REF=4   : num of reference results per runner\n"
        ),
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("xmodel", help="xmodel file path ")
    parser.add_argument(
        "-i", "--subgraph_index", type=int, default=0, help="<subgraph_index>"
    )
    parser.add_argument("num_of_threads", type=int, help="the output png path ")
    parser.set_defaults(func=main)
