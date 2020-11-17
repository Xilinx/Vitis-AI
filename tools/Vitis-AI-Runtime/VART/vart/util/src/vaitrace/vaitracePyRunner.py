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
from subprocess import Popen, PIPE

import collector
import tracer
import writer
import vaitraceCfgManager
import vaitraceSetting
import logging


def pyRunCtx(pyCmd):
    sys.argv[:] = pyCmd
    progname = pyCmd[0]
    sys.path.insert(0, os.path.dirname(progname))

    print("Vaitrace compile python code: %s" % progname)

    with open(progname, 'rb') as fp:
        code = compile(fp.read(), progname, 'exec')

    globs = {
        '__file__': progname,
        '__name__': '__main__',
        '__package__': None,
        '__cached__': None,
    }

    print("Vaitrace exec poython code: %s" % progname)

    exec(code, globs, None)


def run(globalOptions: dict):
    options = globalOptions

    if options.get('cmdline_args').get('bypass', False):
        cmd = options.get('control').get('cmd')
        logging.info("Bypass vaitrace, just run cmd")

        pyCmd = options.get('control').get('cmd')
        pyRunCtx(pyCmd)
        exit(0)

    """Preparing"""
    tracer.prepare(options)
    tracer.start()

    """requirememt format: ["tracerName", "tracerName1", "hwInfo", ...]"""
    collector.prepare(options, tracer.getSourceRequirement())
    collector.start()

    """Start Running"""
    pyCmd = options.get('control').get('cmd')
    timeout = options.get('control').get('timeout')
    pyRunCtx(pyCmd)
    #proc = Popen(cmd)
    #
    #options['control']['pid'] = proc.pid
    #options['control']['time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    options['control']['launcher'] = "python"
    options['control']['pid'] = os.getpid()
    options['control']['time'] = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime())
    #
    # if timeout <= 0:
    #    proc.wait()
    # else:
    #    while timeout > 0:
    #        time.sleep(1)
    #        timeout -= 1
    #        p = proc.poll()
    #        if p is not None:
    #            break
    #
    collector.stop()
    tracer.stop()
    # proc.wait()

    tracer.process(collector.getData())
