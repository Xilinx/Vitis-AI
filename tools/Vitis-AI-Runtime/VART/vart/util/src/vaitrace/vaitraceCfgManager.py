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


import os
import re
import csv
import sys
import string
import time
import argparse
import json
import signal
import logging
from subprocess import Popen, PIPE

import vaitraceDefaults


def merge(a: dict, b: dict):
    if hasattr(a, "keys") and hasattr(b, "keys"):
        for kb in b.keys():
            if kb in a.keys():
                merge(a[kb], b[kb])
            else:
                a.update(b)


def saityCheck(options: dict):
    cmd = options['control']['cmd']
    if cmd == None or len(cmd) == 0:
        logging.error("Wrong command")
        return False

    timeout = options['control']['timeout']
    if timeout < 0:
        logging.error("Wrong timeout")
        return False

    output = options['control']['xat']['filename']
    if output == None or output == "":
        logging.errer("Wrong output file path")
        return False

    xmodel = options.get('xmodel', "")
    if os.path.exists(xmodel) == False:
        logging.debug("Xmodel not exist: [%s]" % xmodel)

    logging.debug("Control Options:")
    logging.debug(options['control'])

    return True


def parseCfg(options: dict):
    try:
        cfgFile = open(options.get('cmdline_args').get('config'), 'rt')
        cfg = json.load(cfgFile)
        overlayOption = cfg.get('options')
    except:
        cfg = options
        overlayOption = {}

    """Configuration priority: Configuration File > Command Line > Default"""
    """Cmd"""
    cmdline_cmd = options['cmdline_args'].get("cmd", "")
    cmd = overlayOption.pop("cmd", cmdline_cmd)
    if cmd == None or len(cmd) == 0:
        logging.error("Can not find excutable command")
        return False

    if type(cmd) == str and cmd != "":
        cmd = cmd.split()
    options['control']['cmd'] = cmd

    """Output"""
    cmdline_output = options['cmdline_args'].get("output", "")
    output = overlayOption.pop("output", cmdline_output)
    if output == "" or output == None:
        shortcmd = options['control']['cmd'][0].split('/')[-1][:15]
        output = os.path.join(os.path.abspath(os.curdir), "%s.xat" % shortcmd)

    options['control']['xat']['filename'] = output

    """Timeout"""
    cmdline_timeout = options['cmdline_args'].get("timeout")
    timeout = overlayOption.pop("timeout", cmdline_timeout)
    options['control']['timeout'] = timeout

    """Vitis Analyzer Mode"""
    """
    For Vitis Analyzer:
    1. do a long trace (100s)
    2. disable [sched] tracer
    3. set env[Debug.xrt_profile] and [Debug.vitis_ai_profile] for xrt
    """
    va_enabled = options['cmdline_args']['va']
    if va_enabled:
        if options['control']['timeout'] == None:
            options['control']['timeout'] = vaitraceDefaults.trace_va_timeout
        merge(options, {'tracer': {'sched': {'disable': True}}})
        os.environ.setdefault("Debug.xrt_profile", "true")
        os.environ.setdefault("Debug.vitis_ai_profile", "true")
    else:
        if options['control']['timeout'] == None:
            options['control']['timeout'] = vaitraceDefaults.trace_xat_timeout

    """Debug"""
    cmdline_debug = options['cmdline_args'].get('debug', False)
    debug = overlayOption.pop("debug", cmdline_debug)
    options['control']['debug'] = debug

    """TraceList"""
    traceList = {}
    traceCfg = cfg.get('trace', {})

    builtInList = vaitraceDefaults.builtInFunctions
    defaultEnabledTraceList = vaitraceDefaults.traceCfgDefaule['trace']["enable_trace_list"]
    enabledTraceList = traceCfg.get('enable_trace_list', defaultEnabledTraceList)
    for trace in enabledTraceList:
        traceName = "trace_" + trace
        if (traceName) in builtInList.keys():
            traceList.update({traceName: builtInList[traceName]})

    traceList.update({"trace_custom": cfg.get("trace_custom", [])})

    merge(options, {'tracer': {'function': {'traceList': traceList}}})

    """Other Options"""
    merge(options, overlayOption)

    return True
