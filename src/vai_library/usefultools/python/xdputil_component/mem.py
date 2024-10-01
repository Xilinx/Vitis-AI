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


import numpy as np
import tools_extra_ops as tools


def mem_read(args):
    np.array(tools.mem_read(args.addr, args.size), dtype=np.uint8).tofile(args.file)


def mem_write(args):
    input_data = np.fromfile(args.file, dtype=np.uint8, count=args.size)
    tools.mem_write(input_data.tolist(), args.addr)


def mem_main(args):
    if args.read:
        mem_read(args)
    elif args.write:
        mem_write(args)


def help(subparsers):
    parser = subparsers.add_parser(
        "mem",
        description="mem ",
        help="<-r|-w> <addr> <size> <output_file|input_file>",
    )
    parser.add_argument(
        "-r",
        "--read",
        help="Read memory, -r or -w must be selected",
        action="store_true",
    )
    parser.add_argument(
        "-w",
        "--write",
        help="Write memory, -r or -w must be selected",
        action="store_true",
    )
    parser.add_argument("addr", type=int, help="Address to read or write data")
    parser.add_argument("size", type=int, help="The size of the data read or written")
    parser.add_argument(
        "file",
        help="When using -r is the output file path, when using -w is the input file path",
    )
    parser.set_defaults(func=mem_main)
