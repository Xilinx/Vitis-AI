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
import struct


def validate_file(file_name):
    if file_name is None:
        return "file name should not be null."
    if not os.path.exists(file_name):
        return file_name + " file doesn't exist."
    if not os.path.isfile(file_name):
        return file_name + " is not a file."
    if os.path.getsize(file_name) % 4:
        return "file size error, not a standard float data bin file."
    return


def main(args):
    # validation check for input golden file
    valid = validate_file(args.golden_file)
    if valid is not None:
        print("golden file check:", valid)
        return
    # validation check for input dump file
    valid = validate_file(args.dump_file)
    if valid is not None:
        print("dump file check:", valid)
        return
    # validation check for threshold
    if args.threshold >= 100:
        print("threshold check: threshold must be less than 100.")
        return

    # get and compare file size
    g_file_size = os.path.getsize(args.golden_file)
    d_file_size = os.path.getsize(args.dump_file)
    if g_file_size != d_file_size:
        print("float bin files have different file size.",
              '\n',
              "golden file and dump file are different!",
              sep='')
        return

    # flag to show more information
    verb = False
    if args.verbose:
        verb = True

    if verb:
        print("float file to compare: ")
        print('\t', "golden file name: " + args.golden_file + ", file size is",
              g_file_size)
        print('\t', "dump file name: " + args.dump_file + ", file size is", d_file_size)
        print('\t', "relative error threshold: " + str(args.threshold) + "%")

    # float number to compare
    floatnum = g_file_size // 4
    filesize = g_file_size
    if verb:
        print("float data number to compare is", floatnum)

    # read bin and convert to float
    with open(args.golden_file, 'br') as gf:
        goldendata = struct.unpack('f' * (floatnum), gf.read(filesize))
    with open(args.dump_file, 'br') as df:
        dumpdata = struct.unpack('f' * (floatnum), df.read(filesize))

    # comparison
    is_diff = False
    diff_index = []
    for idx in range(floatnum):
        if (goldendata[idx] - dumpdata[idx]) == 0.0:
            continue
        if goldendata[idx] == 0.0 or \
           dumpdata[idx] == 0.0 or \
           (abs((goldendata[idx] - dumpdata[idx]) / goldendata[idx]) * 100 > args.threshold):
            is_diff = True
            diff_index.append(idx)

    # result output
    print("float bin file comparison done.")
    if is_diff:
        print("golden file and dump file are different!")
        if verb:
            print("\ndifferent float index list:")
            print("index", "golden", "dump", sep='\t\t')
            for i in diff_index:
                print(i, goldendata[i], dumpdata[i], sep='\t\t')
    else:
        print("golden file and dump file are the same!")


def help(subparsers):
    parser = subparsers.add_parser(
        "comp_float",
        help="<golden_file> <dump_file> [-t threshold] [--verbose]",
    )
    parser.add_argument(
        "golden_file",
        type=str,
        help=
        "the name of input golden bin file, the file will be parsed as float data by every 4 bytes"
    )
    parser.add_argument(
        "dump_file",
        type=str,
        help=
        "the name of input dump bin file, the file will be parsed as float data by every 4 bytes"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.01,
        help=
        "relative error threshold, range [0,100), default 0.01, this threshold should be the percentage multiplied by 100"
    )
    parser.add_argument(
        "--verbose",
        action='store_true',
        help=
        "verbose, if this argument is present, more procedure information will be shown")
    parser.set_defaults(func=main)
