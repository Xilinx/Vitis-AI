#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# Copyright 2019 Xilinx Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import os
import argparse
import logging

import vaitraceCfgManager
import vaitraceSetting
import vaitraceCppRunner
import vaitracePyRunner
import vaitraceDefaults
import vaitraceWriter

VAITRACE_VER = "v1.3-20201026"


def parseCmdLine():
    """
    -o: trace file: save trace file [default: trace.txt]
    -c: conf: input config file
    -t: time: tracing time in second(s)
    -p: python: for tracing python application
    -v: show version
    """

    default_conf_json = ""
    cmd_parser = argparse.ArgumentParser(prog="Xilinx Vitis AI Trace")
    cmd_parser.add_argument("cmd", nargs=argparse.REMAINDER)
    cmd_parser.add_argument("-c", dest="config", nargs='?',
                            default=default_conf_json, help="Specify the config file")
    cmd_parser.add_argument(
        "-d", dest='debug', action='store_true', help="Enable debug")
    cmd_parser.add_argument("-o", dest="traceSaveTo",
                            nargs='?', help="Save trace file to")
    cmd_parser.add_argument("-t", dest='timeout', nargs='?',
                            type=int, help="Tracing time limitation")
    cmd_parser.add_argument("-v", dest='showversion',
                            action='store_true', help="Show version")
    cmd_parser.add_argument("-b", dest='bypass', action='store_true',
                            help="Bypass vaitrace, just run command")
    cmd_parser.add_argument(
        "-p", dest='python', action='store_true', help="Trace python application")
    cmd_parser.add_argument("--va", dest='va', action='store_true',
                            help="Generate trace data for Vitis Analyzer")
    cmd_parser.add_argument(
        "--xat", dest='xat', action='store_true', help="Generate trace data in .xat")

    args = cmd_parser.parse_args()

    if args.showversion:
        print("Xilinx Vitis AI Profiler Tracer Ver %s" % VAITRACE_VER)
        exit(0)

    return args, cmd_parser


def printHelpExit(parser):
    parser.print_help()
    exit(-1)


def main(args, cmd_parser):
    options = vaitraceDefaults.traceCfgDefaule

    """Configuration priority: Configuration File > Command Line > Default"""
    options['cmdline_args'] = {}
    options['cmdline_args']['cmd'] = args.cmd
    options['cmdline_args']['timeout'] = args.timeout
    options['cmdline_args']['output'] = args.traceSaveTo
    options['cmdline_args']['debug'] = args.debug
    options['cmdline_args']['config'] = args.config
    options['cmdline_args']['bypass'] = args.bypass
    options['cmdline_args']['python'] = args.python

    """
    Select Output Format,
    Default format is vtf (or va), only when --xat exists it will generate .xat"""
    if args.xat == True:
        options['cmdline_args']['va'] = False
        options['cmdline_args']['xat'] = True
    else:
        options['cmdline_args']['va'] = True
        options['cmdline_args']['xat'] = False

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.debug("Cmd line args: ")
    logging.debug(args)

    if vaitraceCfgManager.parseCfg(options) == False:
        printHelpExit(cmd_parser)

    vaitraceSetting.setting(options)
    if vaitraceCfgManager.saityCheck(options) == False:
        printHelpExit(cmd_parser)

    if (args.python):
        # Launch Python
        vaitracePyRunner.run(options)
    else:
        # Launch C/C++
        vaitraceCppRunner.run(options)

    vaitraceWriter.write(options)


if __name__ == '__main__':

    """Checking Permission"""
    if os.getgid() != 0:
        logging.error("This tool need run as 'root'")
        exit(-1)

    main(*parseCmdLine())
