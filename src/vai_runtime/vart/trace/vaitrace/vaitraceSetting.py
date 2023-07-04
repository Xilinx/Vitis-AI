
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
import re
import platform
import logging
from vaitraceOptions import merge
import vaitraceDefaults


def getx86CpuTscKhz():
    try:
        cmd = 'cat /proc/cpuinfo | grep "cpu MHz"'
        khz = int(float(os.popen(cmd).readlines()[0].split()[3])) * 1000
    except:
        assert ()
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
    try:
        clocks = open('/sys/kernel/debug/tracing/trace_clock',
                      'rt').read().strip().split()
    except:
        logging.warning(
            "No such file or directory: '/sys/kernel/debug/tracing/trace_clock'")
        logging.warning("CPU function tracing feature is disabled")
        return "boot"

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


def decodeChipIdcode(idcode: str):
    fmlyid = hex(int(idcode, 16) >> 21 & 0x7f)
    sub_fmlyid = hex(int(idcode, 16) >> 18 & 0x7)
    devid = hex(int(idcode, 16) >> 12 & 0x3f)

    if fmlyid == "0x23":
        model = "xlnx,zocl"
    elif fmlyid == "0x26":
        model = "xlnx,zocl-versal"
    else:
        logging.error(f"Unsupported Device, idcode: {idcode}")
        assert (True)
    chip_id = {"idcode": idcode, "fmlyid": fmlyid,
               "sub_fmlyid": sub_fmlyid, "devid": devid, "model": model}
    return chip_id


def getChipId():
    try:
        assert (os.access("/sys/kernel/debug/zynqmp-firmware/pm", os.W_OK) == True)

        pm = open("/sys/kernel/debug/zynqmp-firmware/pm", "wt")
        assert (pm.write("PM_GET_CHIPID") == len("PM_GET_CHIPID"))
        pm.close()

        pm = open("/sys/kernel/debug/zynqmp-firmware/pm", "rt")
        chip_id_raw = pm.read().strip()  # 'Idcode: 0x4724093, Version:0x20000513'
        pm.close()

        idcode = (chip_id_raw.split(',', 1)[0]).split(':', 1)[1].strip()

        return decodeChipIdcode(idcode)

    except:
        logging.error(
            "Cannot get chip id, kernel config CONFIG_ZYNQMP_FIRMWARE_DEBUG=y is required.")
        exit(1)


def getPlatform():
    kRelease = platform.uname().release.split('.')
    machine = platform.uname().machine
    os = platform.uname().system
    model = "unknow"
    plfm = {}
    if machine.startswith("aarch64"):
        plfm = getChipId()

    plfm.update(
        {'os': os, 'release': [kRelease[0], kRelease[1]], 'machine': machine})

    return plfm


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
            logging.warning(
                "[%s] not exists, function tracer disabled" % (cmd))
            return False

    return True


def checkFtrace():
    return os.path.exists("/sys/kernel/debug/tracing/uprobe_events")


def setting(option: dict):
    if (checkEnv() != True) or (checkFtrace() != True):
        """ Disable function tracer """
        disableFunTracer = {'tracer': {'function': {'disable': True}}}
        merge(option, disableFunTracer)

    traceClock = selectTraceClock(option)
    x86_tsc_khz = 0
    if traceClock == 'x86-tsc':
        x86_tsc_khz = getx86CpuTscKhz()
    plat = checkPlatform(getPlatform(), option)

    runmode = option.get("runmode", vaitraceDefaults.default_runmode)
    if runmode.lower() == "debug":
        os.environ.setdefault("XLNX_ENABLE_DEBUG_MODE", "1")
    else:
        runmode = "normal"
    logging.info("VART will run xmodel in [%s] mode" % runmode.upper())

    globalSetting = {'control': {'traceClock': traceClock, 'x86_tsc_khz': x86_tsc_khz,
                                 'platform': plat, 'runmode': runmode}}
    merge(option, globalSetting)
