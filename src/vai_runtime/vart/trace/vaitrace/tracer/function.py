
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

import vaitraceSetting
import tracer.tracerBase
from subprocess import Popen, PIPE
import os

symbolFmt = "%-27s%-35s%-45s%9s\n"
TRACE_NAME_MASK = "__cln2_"

vaiLibPath = ["/usr/lib/"]
vaiLibs = [
    "libxrt_core.so",
    "libxrt_coreutil.so",
    "libvart-buffer-object.so",
    "libvart-dpu-controller.so",
    "libvart-dpu-runner.so",
    "libvart-runner.so",
    "libvart-util.so",
    "libvart-xrt-device-handle.so",

    "libvitis_ai_library-math.so",
    "libvitis_ai_library-general1.so",
    "libvitis_ai_library-dpu_task.so",

    "libvitis_ai_library-tfssd.so",
    "libvitis_ai_library-facerecog.so",
    "libvitis_ai_library-RGBDsegmentation.so",
    "libvitis_ai_library-reidtracker.so",
    "libvitis_ai_library-platerecog.so",
    "libvitis_ai_library-facedetectrecog.so",
    "libvitis_ai_library-pmg.so",
    "libvitis_ai_library-ultrafast.so",
    "libvitis_ai_library-covid19segmentation.so",
    "libvitis_ai_library-3Dsegmentation.so",
    "libvitis_ai_library-facedetect.so",
    "libvitis_ai_library-openpose.so",
    "libvitis_ai_library-facequality5pt.so",
    "libvitis_ai_library-pointpillars.so",
    "libvitis_ai_library-platenum.so",
    "libvitis_ai_library-ssd.so",
    "libvitis_ai_library-yolov2.so",
    "libvitis_ai_library-rcan.so",
    "libvitis_ai_library-solo.so",
    "libvitis_ai_library-platedetect.so",
    "libvitis_ai_library-pointpillars_nuscenes.so",
    "libvitis_ai_library-clocs.so",
    "libvitis_ai_library-pointpainting.so",
    "libvitis_ai_library-facefeature.so",
    "libvitis_ai_library-lanedetect.so",
    "libvitis_ai_library-arflow.so",
    "libvitis_ai_library-multitaskv3.so",
    "libvitis_ai_library-centerpoint.so",
    "libvitis_ai_library-carplaterecog.so",
    "libvitis_ai_library-facelandmark.so",
    "libvitis_ai_library-refinedet.so",
    "libvitis_ai_library-ofa_yolo.so",
    "libvitis_ai_library-ocr.so",
    "libvitis_ai_library-retinaface.so",
    "libvitis_ai_library-vehicleclassification.so",
    "libvitis_ai_library-yolovx.so",
    "libvitis_ai_library-posedetect.so",
    "libvitis_ai_library-medicalsegmentation.so",
    "libvitis_ai_library-bcc.so",
    "libvitis_ai_library-medicalsegcell.so",
    "libvitis_ai_library-hourglass.so",
    "libvitis_ai_library-reid.so",
    "libvitis_ai_library-multitask.so",
    "libvitis_ai_library-efficientdet_d2.so",
    "libvitis_ai_library-textmountain.so",
    "libvitis_ai_library-polypsegmentation.so",
    "libvitis_ai_library-mnistclassification.so",
    "libvitis_ai_library-c2d2_lite.so",
    "libvitis_ai_library-fusion_cnn.so",
    "libvitis_ai_library-medicaldetection.so",

    "libxir.so",
    "libxnnpp-xnnpp.so"
]

"""
traceFuncTable--|--traceFunc--|--prop: top
                |             |--func--|--symbol:  "_Z13run_resnet_50P8dpu_taskRKN2cv3MatENSt7__cxx1112basic_strngI"
                |             |--name--|--name:    "run_resnet_50"
                |                      |--offset:  0x448c
                |                      |--libpath: "/root/lttng/resnet50/test_dnndk_resnet50"
"""


class traceFunc:
    def __init__(self, name, sym, path, offset, section="T"):
        self.name = name
        self.symbol = sym
        self.libPath = path
        self.offset = offset
        self.section = section

    def __str__(self):
        return symbolFmt % \
            (self.name[:25].replace(TRACE_NAME_MASK, '::'),
             self.symbol[:33], self.libPath, hex(self.offset))


class traceSection:
    def __init__(self, _name):
        self.prop = _name
        self.functions = []

    def __str__(self):
        str = "Section: [%s]\n" % self.prop
        str += symbolFmt % ("Function", "SymbolName", "LibPath", "Offset")

        for fun in self.functions:
            str += fun.__str__()

        return str


class symbolTable:
    def __init__(self, _exe):
        self.imgStartFlag = ""
        self.imgEndFlag = ""
        self.exe = os.path.abspath(_exe)
        self.table = []
        self.libs = []
        self.libs.append(self.exe)

        if os.path.exists(self.exe) == False:
            raise RuntimeError("Executable file not exists: [%s]" % self.exe)

    def getSymbols(self, extLibs=""):
        print("Analyzing symbol tables...")

        for libPath in self.libs:

            print("%d / %d" %
                  (self.libs.index(libPath) + 1, len(self.libs)), end="\r")
            if (self.libs.index(libPath) + 1 == len(self.libs)):
                print("")

            res = Popen("nm " + libPath + ' | egrep ".+ [T|t] .+"',
                        shell=True, stderr=PIPE, stdout=PIPE).stdout.readlines()

            symbols = []

            if len(res) > 0:
                """Get symbol tables from nm"""
                for s in res:
                    if len(s.split()) == 3:
                        symbols.append(s.decode().strip('\n').split())
            else:
                """For stripped .so, try get symbols from objdump -T"""
                res = Popen("objdump -T " + libPath,
                            shell=True, stderr=PIPE, stdout=PIPE).stdout.readlines()

                for s in res:
                    if len(s.split()) == 7:
                        deS = s.decode().strip('\n').split()

                        if deS[3] != ".text":
                            continue

                        symbols.append([deS[0], "T", deS[6]])

            self.table.append({"path": libPath, "symbols": symbols})

    def getLibs(self, extExe=""):
        if extExe == "":
            lib = self.exe
        else:
            lib = extExe
            extLibs = []

        lddRes = Popen("ldd " + lib, shell=True, stderr=PIPE,
                       stdout=PIPE).stdout.readlines()

        for line in lddRes:
            line = line.decode().strip('\n').strip('\t')
            index = line.find(" => ")
            if index < 0:
                continue
            libName = line[0:index]
            libPath = line[index+4:]
            libPath = libPath[0:libPath.find(" (")]

            if extExe == "":
                self.libs.append(libPath)
            else:
                extLibs.append(libPath)

        for l1 in vaiLibs:
            match = False
            for l2 in self.libs:
                if l2.find(l1) > 0:
                    match = True
                    break
            if match == False:
                for builtInLibPath in vaiLibPath:
                    if os.path.exists(os.path.join(builtInLibPath, l1)):
                        self.libs.append(os.path.join(builtInLibPath, l1))

        if extExe != "":
            return extLibs

    def getSymbolTable(self):
        self.getLibs()
        self.getSymbols()

        return [lib for lib in self.table if len(lib["symbols"]) > 0]


class traceConf:
    def __init__(self, _traceList, _symbolTable=None, _debug=False):
        self.traceSections = []
        self.symbolTable = _symbolTable
        self.parsed = False
        self.debug = _debug
        self.traceList = _traceList
        self.ftraceSymMap = {}

    def __str__(self):
        if not self.parsed:
            return "un-parsed conf"

        str = "Conf %s: %d section(s)\n" % (self.conf,
                                            len(self.traceSections))
        str += "===========================================================\n"

        for s in self.traceSections:
            str += s.__str__()

        return str

    def matchSymbol(self, _funcName, _table=""):
        if _table == "":
            table = self.symbolTable
        else:
            table = _table

        symbolList = list()
        idx = 0
        searchInLib = ""

        if (_funcName.find("@") > 0):
            searchInLib = _funcName.split('@')[1]
            _funcName = _funcName.split('@')[0]

        for lib in table:
            if (searchInLib != ""):
                if (lib["path"].find(searchInLib) < 0):
                    continue

            for item in lib["symbols"]:
                symbol = item[2]

                """Match C++ function"""
                if symbol.startswith("_Z"):
                    """
                    Match naming space
                    cv::imread -> 2cv6imread
                    """
                    funcName = "".join(["%d%s" % (len(s), s)
                                        for s in _funcName.split('::')])
                else:
                    funcName = _funcName

                """Exact Maching..."""
                em = False
                """_func == _ZN5vitis2ai27classification_post_processERKSt6vectorINS0_7library11InputTensorESaIS3_EERKS1_INS2_12OutputTensorESaIS8_EERKNS0_5proto13DpuModelParamEm"""
                if _funcName.startswith("_Z"):
                    funcName = _funcName
                    em = True

                if (symbol.find(funcName) >= 0) or (em and (symbol == funcName)):
                    offset = int(item[0], 16)
                    sym = symbol
                    path = lib["path"]

                    if em:
                        try:
                            showName = os.popen(
                                "c++filt -p %s" % _funcName).readlines()[0].strip()
                        except:
                            showName = _funcName
                    else:
                        if idx > 0:
                            showName = _funcName + "_%d" % idx
                        else:
                            showName = _funcName

                    ftraceSymName = str(
                        "_fun_" + str(hash(showName) % 10000000))
                    self.ftraceSymMap.update({ftraceSymName: showName})
                    self.ftraceSymMap.update({showName: ftraceSymName})

                    idx += 1

                    """Origin name for show"""
                    #symbolList.append(traceFunc(_funcName.split("::")[-1], sym, path, offset))
                    symbolList.append(
                        traceFunc(ftraceSymName, sym, path, offset))

        return symbolList

    def parseTableJson(self):
        if self.parsed == True:
            return self.traceSections

        conf = self.traceList

        """Add and match functions to be traced get keys that starts with 'trace'"""
        for section in [s for s in conf.keys() if s.startswith('trace')]:
            if conf[section] is None:
                continue

            """Remove duplicates"""
            conf[section] = list(set(conf[section]))

            sec = traceSection(section)

            for func in conf[section]:
                print("%d / %d" %
                      (conf[section].index(func) + 1, len(conf[section])), end="\r")

                if conf[section].index(func) + 1 == len(conf[section]):
                    print("")

                match = self.matchSymbol(func, self.symbolTable)

                if len(match) > 0:
                    for s in match:
                        sec.functions.append(s)
                else:
                    if self.debug:
                        ask("Function [%s] not found in symbol table" % func)

            self.traceSections.append(sec)

        self.parsed = True
        return self.traceSections


class functionTracer(tracer.tracerBase.Tracer):
    def __init__(self):
        super().__init__('function', source=["ftrace"], compatible={
            'machine': ["x86_64", "aarch64"]})
        self.ftraceSymMap = {}

    def prepare(self, options: dict, debug: bool):
        """Process configure file & symbol table"""
        table = symbolTable(options.get(
            'control').get('cmd')[0]).getSymbolTable()
        optionFunction = options.get('tracer', {}).get('function', {})
        traceListOri = optionFunction.get('traceList')

        z = traceConf(traceListOri, table)

        traceList = []
        traceListOri = z.parseTableJson()
        self.ftraceSymMap = z.ftraceSymMap

        "Handle Input Options"

        for section in traceListOri:
            if not section.prop.startswith("trace_"):
                continue
            for func in section.functions:
                traceList.append(
                    [func.name, func.libPath, "0x%x" % func.offset])

        saveTo = None
        if debug:
            saveTo = './function.trace'

        "Handle Output Options"
        optForFunction = {
            "collector": {
                "ftrace": {
                    "function": {
                        "name": "function",
                        "type": "uprobe",
                        "saveTo": saveTo,
                        "traceList": traceList,
                        "ftraceSymMap": self.ftraceSymMap
                    }
                }
            }
        }

        return optForFunction

    def compatible(self, platform: {}):
        if super().compatible(platform) == False:
            return False

        """Do some tests"""
        return vaitraceSetting.checkFtrace()

    def process(self, data, t_range=[]):
        """"test_performanc-19925 [003] .... 110337.353720: _fun_9351538_entry: (0xffff8553b0d8)"""
        data = [l for l in data.get('ftrace', {}).get(
            self.name) if not l.startswith('#')]
        data_out = []
        for l in data:
            ftraceSym = l.strip().split()[4].rsplit('_', 1)[0]
            data_out.append(l.replace(ftraceSym, self.ftraceSymMap[ftraceSym]))
            # for k in self.ftraceSymMap.keys():
            #    if l.find(k) > 0:
            #        data_out.append(l.replace(k, self.ftraceSymMap[k]))
        self.data = data_out

    def getData(self):
        return self.data


tracer.tracerBase.register(functionTracer())
