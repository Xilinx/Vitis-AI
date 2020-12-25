
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
import platform
import logging
from vaitraceOptions import merge


def getx86CpuTscKhz():
    try:
        cmd = 'cat /proc/cpuinfo | grep "cpu MHz"'
        khz = int(float(os.popen(cmd).readlines()[0].split()[3])) * 1000
    except:
        assert()
    return khz


"""
+--------------+-------------------+------------+
| clock_source | x86               | arm        |
+--------------+-------------------+------------+
| VA           | global + XRT      | boot + XRT |
| XAT          | x86-tsc + x86-tsc | boot + XRT |
+--------------+-------------------+------------+
"""
def selectTraceClock(option):
    clocks = open('/sys/kernel/debug/tracing/trace_clock',
                  'rt').read().strip().split()

    va_enabled = option['cmdline_args']['va']
    if va_enabled:
        """VTF do not accept x86-tsc"""
        for c in clocks:
            if c.find("x86-tsc") > 0:
                clocks.remove(c)
                break

    """Ranked by priority"""
    #preferClocks = ["boot", "x86-tsc", "global"]
    preferClocks = ["boot", "global"]
    traceClock = "global"

    for pC in preferClocks:
        for c in clocks:
            if c.find(pC) >= 0:
                traceClock = pC
                break
        else:
            continue
        break

    logging.debug("Use %s as trace clock" % traceClock)

    return traceClock


def getPlatform():
    kRelease = platform.uname().release.split('.')
    machine = platform.uname().machine
    return {'release': [kRelease[0], kRelease[1]], 'machine': machine}

def checkPlatform(plat, option):
    xat_enabled = option['cmdline_args']['xat']
    if xat_enabled and plat.get('machine') == 'x86_64':
        logging.error(".xat format not available for cloud platforms")
        exit(-1)
    return plat


def checkEnv():
    cmds = ["nm --version", "ldd --version", "objdump --version"]

    def checkCmd(cmd):
        try:
            ret = os.system("%s > /dev/null" % cmd)
            if ret != 0:
                return False
        except:
            return False

        return True

    for cmd in cmds:
        if checkCmd(cmd) == False:
            logging.error("[%s] not exists, please check the document" % (cmd))
            return False

    return True


def setting(option: dict):
    if checkEnv() != True:
        exit(-1)

    traceClock = selectTraceClock(option)
    x86_tsc_khz = 0
    if traceClock == 'x86-tsc':
        x86_tsc_khz = getx86CpuTscKhz()
    plat = checkPlatform(getPlatform(), option)

    runmode = option.get("runmode", "normal")
    if runmode == "debug":
        os.environ.setdefault("XLNX_ENABLE_DEBUG_MODE", "1")
    logging.info("VART will run xmodel in [%s] mode" % runmode.upper())

    globalSetting = {'control': {'traceClock': traceClock, 'x86_tsc_khz': x86_tsc_khz,
                                 'platform': plat, 'runmode': runmode}}
    merge(option, globalSetting)
