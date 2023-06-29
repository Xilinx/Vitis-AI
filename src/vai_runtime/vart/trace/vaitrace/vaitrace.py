#!/usr/bin/python3
# -*- coding: UTF-8 -*-

# Copyright 2022-2023 Advanced Micro Devices Inc.

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


def getVersion():
    head = "vaitrace ver:\n"
    version = ""
    project = ""
    internal_version = ""
    try:
        import vaitrace_version as v
        version = "%s_%s_%s\nProject:\n%s" % \
            (v.VAITRACE_VER, v.VAITRACE_GIT_VER,
             v.VAITRACE_BUILD_DATE, v.TOP_PROJECT)
    except:
        path = os.path.realpath(__file__)
        version = "Dev Version: [%s]" % path
        return version

    try:
        git_inter_prev_ver = v.VAITRACE_GIT_INTERNAL_PREV_VER
    except:
        git_inter_prev_ver = "-"

    try:
        git_inter_ver = v.VAITRACE_INTERNAL_VER
    except:
        git_inter_ver = "-"

    internal_version = "\nInternal Git Ver:\n[%s, %s]" % (
        git_inter_prev_ver, git_inter_ver)

    return head + version + internal_version


def checkPermisson():
    """Checking Permission"""
    if os.getgid() != 0:
        logging.warning("This tool need run as 'root'")
        logging.warning(
            "without 'root' permission, CPU function profiling feature is invalid")


def parseCmdLine():
    """
    -o: trace file: save trace file [default: trace.txt]
    -c: conf: input config file
    -t: time: tracing time in second(s)
    -p: python: for tracing python application
    -v: show version
    """

    default_conf_json = ""
    cmd_parser = argparse.ArgumentParser(prog="vaitrace")
    cmd_parser.add_argument("cmd", nargs=argparse.REMAINDER)
    cmd_parser.add_argument("-c", dest="config", nargs='?',
                            default=default_conf_json, help="Specify the config file")
    cmd_parser.add_argument(
        "-d", dest='debug', action='store_true', help="Enable debug")
    cmd_parser.add_argument("-o", dest="traceSaveTo",
                            nargs='?', help="Save report to, only available for txt summary mode")
    cmd_parser.add_argument("-t", dest='timeout', nargs='?',
                            type=int, help="Tracing time limit in second, default value is 60")
    cmd_parser.add_argument("-v", dest='showversion',
                            action='store_true', help="Show version")
    cmd_parser.add_argument("-b", dest='bypass', action='store_true',
                            help="Bypass vaitrace, just run command")
    cmd_parser.add_argument(
        "-p", dest='python', action='store_true', help="Trace python application")
    cmd_parser.add_argument("--va", dest='va', action='store_true',
                            help="Generate trace data for Vitis Analyzer")
    cmd_parser.add_argument(
        "--xat", dest='xat', action='store_true', help="Save raw data, for debug usage")
    cmd_parser.add_argument(
        "--txt_summary", dest='txt', action='store_true', help="Display txt summary")
    cmd_parser.add_argument(
        "--json_summary", dest='json', action='store_true', help="Display json summary")
    # fine_grained feature was removed by xcompiler since vai-3.0
    #cmd_parser.add_argument(
    #    "--fine_grained", dest='fg', action='store_true', help="Fine grained mode")

    args = cmd_parser.parse_args()

    if args.showversion:
        print(getVersion())
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
        options['cmdline_args']['xat'] = True
        options['cmdline_args']['va'] = False
        options['cmdline_args']['txt'] = False
        options['cmdline_args']['json'] = False
    elif args.txt == True:
        options['cmdline_args']['txt'] = True
        options['cmdline_args']['xat'] = False
        options['cmdline_args']['va'] = True
        options['cmdline_args']['json'] = False
    elif args.json == True:
        options['cmdline_args']['json'] = True
        options['cmdline_args']['xat'] = False
        options['cmdline_args']['va'] = True
        options['cmdline_args']['txt'] = False
    else:
        options['cmdline_args']['va'] = True
        options['cmdline_args']['xat'] = False
        options['cmdline_args']['txt'] = False

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    """
    if args.fg:
        logging.warning("The fine grained profiling model is enabled. Models will be run subgraph-by-subgraph.\n"
                        "1. It will introduce significant scheduling and memory access overhead, especially for small models.\n"
                        "2. Some xcompiler's optimizations will be turned-off.\n")
        options['runmode'] = 'debug'
        options['cmdline_args']['fg'] = True
    """
    logging.debug("Command line args: ")
    logging.debug(args)

    if vaitraceCfgManager.parseCfg(options) == False:
        printHelpExit(cmd_parser)

    vaitraceSetting.setting(options)
    if vaitraceCfgManager.saityCheck(options) == False:
        printHelpExit(cmd_parser)

    if (args.python):
        checkPermisson()
        # Launch Python
        vaitracePyRunner.run(options)
    else:
        checkPermisson()
        # Launch C/C++
        vaitraceCppRunner.run(options)

    vaitraceWriter.write(options)


if __name__ == '__main__':

    main(*parseCmdLine())
