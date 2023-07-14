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
import re
import csv
import sys
import string
import time
import argparse
import json
import signal
import platform
import gzip
import pickle
import time
import logging
from subprocess import Popen, PIPE

import tracer
import writer
import vaitraceCfgManager
import vaitraceSetting

import writer.vtf.convert
import writer.txt.convert
import writer.json.convert


def dumpRaw(data, saveTo, compress=False):
    if compress:
        xat_file = gzip.open(saveTo, 'wb+')
    else:
        xat_file = open(saveTo, 'wb+')

    pickle.dump(data, xat_file)

    logging.info(".xat file Saved to [%s]" % os.path.abspath(saveTo))
    xat_file.close()


def write(globalOptions: dict):
    options = globalOptions
    debug = options.get('cmdline_args', {}).get('debug', False)

    rawData = tracer.getData()
    outFmt = 'xat'
    va_enabled = options.get('cmdline_args', {}).get('va', False)
    if va_enabled:
        outFmt = 'vtf'

    txt_enabled = options.get('cmdline_args', {}).get('txt', False)
    if txt_enabled:
        """Override outfmt to txt"""
        outFmt = 'txt'

    json_enabled = options.get('cmdline_args', {}).get('json', False)
    if json_enabled:
        """Override outfmt to json"""
        outFmt = 'json'

    json_enabled = options.get('cmdline_args', {}).get('json', False)
    if json_enabled:
        """Override outfmt to json"""
        outFmt = 'json'

    if (debug):
        dumpRaw(rawData, "./vtf_debug.xat", True)

    if outFmt == 'xat':
        """Dump Data"""
        saveTo = options.get('control').get('xat').get('filename')
        compress = options.get('control').get('xat').get('compress', True)

        dumpRaw(rawData, saveTo, compress)

    elif outFmt == 'vtf':
        logging.info("Generating VTF")
        writer.vtf.convert.xat_to_vtf(rawData, options)

    elif outFmt == 'txt':
        logging.info("Generating ascii-table summary")
        writer.txt.convert.xat_to_txt(rawData, options)

    elif outFmt == 'json':
        logging.info("Generating json summary")
        writer.json.convert.xat_to_json(rawData, options)

    else:
        logging.error("Undefined format")
        exit(-1)


if __name__ == "__main__":
    input_xat = sys.argv[1]
    if len(sys.argv) == 3:
        output_path = sys.argv[2]
    elif len(sys.argv) == 2:
        output_path = os.path.dirname(input_xat)

    f = gzip.open(input_xat, 'rb')
    xat = pickle.load(f)

    #writer.vtf.convert.xat_to_vtf(xat, output_path)
    writer.txt.convert.xat_to_txt(xat)
