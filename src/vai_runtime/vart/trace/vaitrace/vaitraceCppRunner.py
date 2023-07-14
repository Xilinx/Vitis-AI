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


import os
import sys
import time
import signal
import time
import logging
from subprocess import Popen, PIPE

import collector
import tracer
import vaitraceCfgManager
import vaitraceSetting


force_exit = False


def handler(signum, frame):
    logging.info("Killing process...")
    logging.info("Processing trace data, please wait...")
    global force_exit
    if force_exit:
        logging.error("Force exit...")
        exit(-1)
    force_exit = True


def shell_find_exec_path(_exe):

    exe_abs_path = ""

    if os.path.exists(os.path.abspath(_exe)):
        """1. search in cur dir"""
        exe_abs_path = os.path.abspath(_exe)

    else:
        """2. search in path via 'which'"""
        which_cmd = ["which", _exe]
        p = Popen(which_cmd, stdout=PIPE, stderr=PIPE)
        res = p.stdout.readlines()
        if len(res) > 0:
            exe_abs_path = res[0].strip().decode()

    if os.path.exists(exe_abs_path) == False:
        raise RuntimeError("Executable file not exists [%s]" % _exe)

    logging.info("Executable file: %s" % exe_abs_path)

    return exe_abs_path


def run(globalOptions: dict):
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    options = globalOptions

    """Help find the path of cmd in PATH"""
    cmd = options.get('control').get('cmd')[0]
    options['control']['cmd'][0] = shell_find_exec_path(cmd)

    if options.get('cmdline_args').get('bypass', False):
        cmd = options.get('control').get('cmd')
        logging.info("Bypass vaitrace, just run cmd")
        proc = Popen(cmd)
        proc.wait()
        exit(0)

    """Preparing"""
    tracer.prepare(options)
    tracer.start()

    """requirememt format: ["tracerName", "tracerName1", "hwInfo", ...]"""
    collector.prepare(options, tracer.getSourceRequirement())
    collector.start()

    """Start Running"""
    cmd = options.get('control').get('cmd')
    timeout = options.get('control').get('timeout')
    proc = Popen(cmd)

    options['control']['launcher'] = "cpp"
    options['control']['pid'] = proc.pid
    options['control']['time'] = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime())

    if timeout <= 0:
        proc.wait()
    else:
        while timeout > 0:
            time.sleep(1)
            timeout -= 1
            p = proc.poll()
            if p is not None:
                break

    if (timeout == 0):
        logging.info("vaitrace time out, stopping process...")
        proc.send_signal(signal.SIGINT)
    proc.wait()
    collector.stop()
    tracer.stop()

    tracer.process(collector.getData())
