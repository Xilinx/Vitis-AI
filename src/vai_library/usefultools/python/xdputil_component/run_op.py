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
import os
from typing import List
import xir
import vart
import tools_extra_ops as tools


def main(args):
    if args.ref_dir is None:
        args.ref_dir = os.getcwd() + "/ref"

    if os.path.exists(args.ref_dir) and not os.path.isdir(args.ref_dir):
        print(args.ref_dir + " should be a directory.")
        return
    elif not os.path.exists(args.ref_dir):
        os.mkdir(args.ref_dir)
        print("default directory " + args.ref_dir +
              " has been created, please put the input tensor file in.")
        return

    if args.dump_dir is None:
        args.dump_dir = os.getcwd() + "/dump"

    if os.path.exists(args.dump_dir) and not os.path.isdir(args.dump_dir):
        print(args.dump_dir + " should be a directory.")
        return
    elif not os.path.exists(args.dump_dir):
        os.mkdir(args.dump_dir)
        print("default directory " + args.dump_dir + " has been created.")

    if tools.test_op_run(args.xmodel, args.op_name, args.ref_dir,
                         args.dump_dir):
        print("test pass")
    else:
        print("test failed")


def help(subparsers):
    parser = subparsers.add_parser(
        "run_op",
        help="<xmodel> <op_name> [-r ref_dir] [-d dump_dir]",
    )
    parser.add_argument("xmodel", type=str, help="xmodel file name")
    parser.add_argument(
        "op_name",
        type=str,
        help=
        "op name, this op_name should be consistent with the name in xmodel")
    parser.add_argument(
        "-r",
        "--ref_dir",
        type=str,
        help=
        "reference directory, this directory default as \"ref\" should contain inputs tensor file like <TENSOR_NAME>.bin"
    )
    parser.add_argument(
        "-d",
        "--dump_dir",
        type=str,
        help=
        "dump directory, this directory default as \"dump\" will be the dump destination of output tensor file"
    )
    parser.set_defaults(func=main)
