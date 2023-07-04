
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

import collector
import platform
import vaitraceSetting
import os
import logging
import time

FTRACE_DIR = "/sys/kernel/debug/tracing/"


class Ftrace:
    def __init__(self, _name, _path=FTRACE_DIR, _buffersize=8192, _type=None, _clk='local', ftraceSymMap={}, saveTo=None, _debug=False):
        self.cwd = os.getcwd()
        self.instPath = ""
        self.debug = _debug
        self.path = _path
        self.traceHandler = None
        self.saveTo = saveTo
        self.ftraceSymMap = ftraceSymMap
        self.name = _name
        self.traceClock = _clk
        self.traceType = _type
        self.quirk = False

        if vaitraceSetting.checkFtrace() == False:
            logging.error("Ftrace not Enable")

        if self.checkKernelRelease()[0] < 4:
            self.legacyKernel = True
        else:
            self.legacyKernel = False
        if (self.legacyKernel) and (self.traceType == 'uprobe'):
            self.quirk = True

        self.buffersize = _buffersize
        self.createInstance(_name, _path)
        self.setBufferSize(_buffersize)
        self.setClockSource()

    def __del__(self):
        """Delete ftrace instance"""
        os.system("rmdir %s" % self.instPath)

        self.disableAllEvents()
        self.clearUprobe()
        self.clearKprobe()

    def backCwd(self):
        if self.cwd == "":
            raise IOError()
        os.chdir(self.cwd)

    def createInstance(self, _name, path):
        # if self.legacyKernel == True and self.traceType == 'user':
        if False:
            """
            For Linux Kernel-3.x, do not use [instance],
            just create trace points in /sys/kernel/debug/tracing dir
            """
            self.name = _name
            os.chdir(path)
            self.instPath = os.getcwd()
            self.backCwd()
        else:
            # if self.legacyKernel == False:
            try:
                """Create ftrace instance"""
                os.chdir(os.path.join(path, "instances"))
                os.mkdir(_name)
            except IOError:
                if os.path.exists(_name) and os.path.isdir(_name):
                    pass
                else:
                    raise IOError("Create instance failed")
            finally:
                os.chdir(_name)
                self.name = _name
                self.instPath = os.getcwd()
                self.backCwd()

    def checkKernelRelease(self):
        kRelease = platform.uname().release.split('.')
        return (int(kRelease[0]), int(kRelease[1]))

    def _toTraceInstDir(func):
        def wrapper(self, *args, **kw):
            backInst = (os.getcwd() == self.instPath)

            os.chdir(self.instPath)
            ret = func(self, *args, **kw)
            if not backInst:
                self.backCwd()
            return ret
        return wrapper

    def _toTraceDir(func):
        def wrapper(self, *args, **kw):
            os.chdir(self.path)
            ret = func(self, *args, **kw)
            self.backCwd()
            return ret
        return wrapper

    @_toTraceInstDir
    def setBufferSize(self, bufsize):
        if self.quirk:
            os.chdir(self.path)
        open("buffer_size_kb", 'w').write(str(bufsize))

    @_toTraceInstDir
    def setClockSource(self):
        traceClock = self.traceClock
        open("trace_clock", 'w').write(traceClock)

    @_toTraceInstDir
    def enableTracing(self):
        if self.quirk:
            os.chdir(self.path)
        logging.debug("### enable tracing %s" % os.getcwd())
        open("tracing_on", 'w').write("1")

    @_toTraceInstDir
    def disableTracing(self):
        if self.quirk:
            os.chdir(self.path)
        logging.debug("### disable tracing %s" % os.getcwd())
        open("tracing_on", 'w').write("0")

    @_toTraceInstDir
    def enableEvent(self, system, event, filter=""):
        """
        For example, system might be [sched] and event [sched_switch]
        Then we should write 1 to ./events/[sched]/[sched_switch]/enable
        echo '(prev_comm == test_dnndk_resn) || (next_comm == test_dnndk_resn)' > ./events/sched/sched_switch/filter
        echo 1 > ./events/sched/sched_switch/enable
        """
        if (filter != ""):
            path = os.path.join("./", "events", system, event, "filter")
            open(path, 'w').write(filter)

        path = os.path.join("./", "events", system, event, "enable")
        if os.path.exists(path) == True:
            open(path, 'w').write("1")

    @_toTraceInstDir
    def disableEvent(self, system, event):
        """
        For example, system might be [sched] and event [sched_switch]
        Then we should write 0 to ./events/[sched]/[sched_switch]/enable
        echo 0 > ./events/sched/sched_switch/enable
        """
        path = os.path.join("./", "events", system, event, "enable")
        if os.path.exists(path) == True:
            open(path, 'w').write("0")

    @_toTraceDir
    def addUprobe(self, func, lib, offset):
        """
        This func will add uprobe and uretprobe event at the same time
        lib should be abspath
        example:
        echo 'p:runResnet50_entry /root/lttng/resnet50/test_dnndk_resnet50:0x448c' >> /t/uprobe_events
        echo 'r:runResnet50_exit /root/lttng/resnet50/test_dnndk_resnet50:0x448c' >> /t/uprobe_events
        """

        os.chdir(self.path)
        """uprobe_event can accept symbol length <= 64 characters"""
        limit = 64 - len("_entry")
        if len(func) >= limit:
            logging.debug("Too long symbol name [%s]" % (func))
            func = func[::-1][:limit:][::-1]

        cmdU = "p:" + func + "_entry " + lib + ":" + offset
        cmdUret = "r:" + func + "_exit " + lib + ":" + offset

        logging.debug("### uprobe %s" % cmdU)
        if self.debug:
            print("### uprobe %s" % cmdU)
            print("### uretprobe %s" % cmdUret)

        # open failed ?????
        #open("./uprobe_events", "a+").writeline(cmdU)
        #open("./uprobe_events", "a+").writeline(cmdUret)

        ret = os.system("echo %s >> ./uprobe_events" % cmdU)
        if ret != 0:
            logging.warning("[%s] event un-match" % cmdU)
            return
        ret = os.system("echo %s >> ./uprobe_events" % cmdUret)
        if ret != 0:
            logging.warning("[%s] event un-match" % cmdUret)

        self.enableUprobe(func)
        #print(open("./uprobe_events", "r").read())

    @_toTraceDir
    def addKprobe(self, name, mod, offset, fetchargs: []):
        """
        This func will add kprobe event
        lib should be abspath
        example:
        echo 'p:cu_start zocl:zocl_hls_start cu_idx=+0(%x0)' >> kprobe_events
        echo 'p:cu_done zocl:zocl_hls_check+0x64 cu_idx=+0(%x21)' >> kprobe_events
        """

        os.chdir(self.path)
        cmd = "p:%s %s:%s" % (name, mod, offset)
        for arg in fetchargs:
            cmd += " %s" % arg

        if self.debug:
            logging.debug("### kprobe %s" % cmd)

        os.system("""echo '%s' >> ./kprobe_events""" % cmd)

        self.enableKprobe(name)

    @_toTraceDir
    def clearUprobe(self):
        if self.debug:
            logging.debug("Clear all Uprobe & Uretprobe events")

        # open failed ?????
        #open("./uprobe_events", "a+").writeline(cmdU)
        #open("./uprobe_events", "a+").writeline(cmdUret)

        #print("echo  > ./uprobe_events", self.instPath)
        os.system("echo  > ./uprobe_events")

    @_toTraceDir
    def clearKprobe(self):
        if self.debug:
            logging.debug("Clear all Kprobe & Kretprobe events")

        # open failed ?????
        #open("./uprobe_events", "a+").writeline(cmdU)
        #open("./uprobe_events", "a+").writeline(cmdUret)

        #print("echo  > ./kprobe_events", self.instPath)
        os.system("echo  > ./kprobe_events")

    @_toTraceDir
    def disableAllEvents(self):
        if self.debug:
            print("Disable all events")

        # open failed ?????
        #open("./uprobe_events", "a+").writeline(cmdU)
        #open("./uprobe_events", "a+").writeline(cmdUret)

        os.system("echo 0 > ./events/enable")
        time.sleep(0.1)

    @_toTraceInstDir
    def enableUprobe(self, func):
        self.enableEvent("uprobes", func+"_entry")
        self.enableEvent("uprobes", func+"_exit")

    @_toTraceInstDir
    def enableKprobe(self, probePoint):
        self.enableEvent("kprobes", probePoint)

    @_toTraceInstDir
    def disableUprobe(self, func):
        self.disableEvent("uprobes", func+"_entry")
        self.disableEvent("uprobes", func+"_exit")

    @_toTraceInstDir
    def getTrace(self):
        # if self.legacyKernel == True and self.traceType == 'uprobe':
        if self.quirk:
            os.chdir(self.path)

        trace = open("trace", "r").readlines()

        if self.saveTo != None:
            data_out = []
            for l in trace:
                uprobeFnMatched = False
                matched = False
                for k in [_k for _k in self.ftraceSymMap.keys() if _k.startswith("_fun_")]:
                    if l.find(k) > 0:
                        data_out.append(l.replace(k, self.ftraceSymMap[k]))
                        matched = True
                        break
                if matched == False:
                    data_out.append(l)

            f = open(self.saveTo, "wt+")
            logging.debug("Trace file was saved to [%s]" % self.saveTo)
            f.writelines(data_out)
            f.close()

        return trace

    @_toTraceInstDir
    def startTracePipe(self):
        self.traceHandler = Popen(
            ["cat", "trace_pipe"], stdout=PIPE, stderr=PIPE)

    @_toTraceInstDir
    def stopTracePipe(self, saveTo=""):
        t = self.traceHandler
        t.terminate()

        trace = t.stdout.readlines()

        if saveTo != "":
            f = gzip.open(saveTo, "wt")
            f.writelines(trace)
            f.close()

        return trace

    @_toTraceInstDir
    def clearTracing(self, disable=True):
        if disable:
            self.disableTracing()
        open("trace", 'w').write('')


class ftraceCollector(collector.collectorBase.Collector):
    def __init__(self):
        super().__init__(name='ftrace')
        self.instances = []

    def __del__(self):
        pass

    def prepare(self, conf: dict) -> dict:
        """
        type: "kprobe" | "uprobe" | "event"
        list for kprobe: [name, mod, offset, fetchargs: []]
        list for uprobe: [func, lib, offset]
        list for event : [system, event, filter=""]

        """

        ftraceOption = conf.get('collector', {}).get('ftrace')
        if ftraceOption == None:
            return conf

        traceClock = conf.get('control', {}).get('traceClock', 'global')
        open(os.path.join(FTRACE_DIR, "trace_clock"), 'w').write(traceClock)

        for name in ftraceOption.keys():
            instOption = ftraceOption[name]
            traceType = instOption.get('type')
            if instOption.get('name', None) != name:
                assert ()

            _saveTo = instOption.get('saveTo', None)
            self.ftraceSymMap = instOption.get("ftraceSymMap", {})

            if _saveTo != None:
                _saveTo = os.path.abspath(_saveTo)
            traceInst = Ftrace(name, ftraceSymMap=self.ftraceSymMap, saveTo=_saveTo,
                               _clk=traceClock, _type=traceType)

            if traceType == 'kprobe':
                # dpu.addKprobe("cu_start", "zocl", "zocl_hls_start", ["cu_idx=+0(%x0):u32"])
                for t in instOption.get("traceList", None):
                    traceInst.addKprobe(*t)
            elif traceType == 'uprobe':
                # addUprobe(func.name, func.libPath, "0x%x" % func.offset)
                for t in instOption.get("traceList", None):
                    traceInst.addUprobe(*t)
            elif traceType == 'event':
                for t in instOption.get("traceList", None):
                    traceInst.enableEvent(*t)
            else:
                assert ()

            self.instances.append(traceInst)

        for ft in self.instances:
            ft.clearTracing()

        return conf

    def start(self):
        super().start()
        logging.debug("### ftrace start")

        for inst in self.instances:
            inst.enableTracing()

    def stop(self):
        super().stop()

        logging.debug("### ftrace stop")

        for inst in self.instances:
            inst.disableTracing()

    """
    data = {
        ftrace: {
            cuEdge: xxxxxx,
            sched: xxxxxx,
        }
    }
    
    """

    def getData(self):
        ftraceData = {}

        for inst in self.instances:
            ftraceData.update({inst.name: inst.getTrace()})

        return ftraceData


collector.collectorBase.register(ftraceCollector())
