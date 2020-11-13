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

import parser.parserBase
from parser.DPUController import subGraphStat
import json


class xirTensor:
    def __init__(self):
        pass


class xirKernel:
    def __init__(self):
        pass


class xirGraph():
    def __init__(self, _fineGrained=False):
        self.tensors = dict()
        self.links = list()
        self.kernels = list()
        self.nodes = list()
        self.graph = dict()
        self.fineGrained = _fineGrained

    def addGraph(self, _graph):
        for k in _graph:
            if len(k) == 0:
                _graph.remove(k)
                continue
            for s in k.get('subgraph'):
                if len(s) == 0:
                    k.get('subgraph').remove(s)
        self.graph = _graph

        idx = 1
        for kernel in self.graph:
            self.addKernel(kernel, idx)
            idx += 1

    def addTensor(self, _name, node, dir):
        if _name not in self.tensors:
            self.tensors.update({_name: {'in': set(), 'out': set()}})
        t = self.tensors[_name]
        if dir == 'in':
            t['in'].add(node)
        elif dir == 'out':
            t['out'].add(node)

    def addKernel(self, kernel, idx):
        if len(kernel) == 0:
            return

        id = idx
        name = kernel.get('kernelName')
        dev = kernel.get('kernelDev')
        subGraphs = kernel.get('subgraph')

        for inTensor in kernel.get('inputs'):
            if len(inTensor) == 0:
                continue
            self.addTensor(inTensor, "SubGraph-%d" % id, 'in')
        for outTensor in kernel['outputs']:
            if len(outTensor) == 0:
                continue
            self.addTensor(outTensor, "SubGraph-%d" % id, 'out')

        self.kernels.append({
            'name': name,
            'dev': dev,
            'id': id,
            'subgraph': subGraphs
        })

    def makeLinks(self):
        for tensorName in self.tensors:
            tensor = self.tensors[tensorName]
            inNodes = tensor['in']
            outNodes = tensor['out']
            if len(inNodes) == 0 or len(outNodes) == 0:
                continue
            for linkE in inNodes:
                for linkS in outNodes:
                    self.links.append({'source': linkS, 'target': linkE})

        return self.links

    def makeTreeGridTable(self):
        treeGridTable = []
        subGraphStatTable = {}

        for sub in subGraphStat:
            subgTime = subGraphStat[sub]
            minTus = min(subgTime) * 1000 * 1000
            maxTus = max(subgTime) * 1000 * 1000
            meanTus = sum(subgTime) / len(subgTime) * 1000 * 1000
            subGraphStatTable.update({sub: [minTus, maxTus, meanTus]})

        for k in self.graph:
            #kernelKey = k.get('kernelName').split('(')[0]
            kernelKey = k.get('kernelName', "")
            kernelKeyShort = kernelKey.replace('/', '_').split('(')[0]
            kernelT = 0

            if not self.fineGrained:
                if kernelKeyShort in subGraphStatTable.keys():
                    kernelT = subGraphStatTable[kernelKeyShort][2]

            kernel = {
                'title': k.get('kernelName'),
                'device': k.get('kernelDev'),
                'key': kernelKey,
                'workload': 0,
                'time': kernelT,
                'folder': True,
                'icon': False,
                'children': []
            }

            for s in k.get('subgraph'):
                if not self.fineGrained:
                    break
                subgraphKey = s.replace('/', '_').split('(')[0]
                subGraphT = 0
                if subgraphKey in subGraphStatTable.keys():
                    subGraphT = subGraphStatTable[subgraphKey][2]

                subGraph = {
                    'title': s,
                    'device': k.get('kernelDev'),
                    'key': "0",
                    'workload': 0,
                    'time': subGraphT,
                    'folder': True,
                    'icon': False,
                }
                kernel.get('children').append(subGraph)

            treeGridTable.append(kernel)

        return treeGridTable

    def makeNodes(self):
        x_offset_user = 0
        x_offset_dpu = 150
        x_offset_cpu = 300
        y_offset = 60
        nodes = []
        head = [{
            'name': 'User Kernels',
            'category': 'user',
            'x': x_offset_user,
            'y': 0
        }, {
            'name': 'DPU Kernels',
            'category': 'dpu',
            'x': x_offset_dpu,
            'y': 0
        }, {
            'name': 'CPU Kernels',
            'category': 'cpu',
            'x': x_offset_cpu,
            'y': 0
        }]

        for kernel in self.kernels:
            dev = kernel.get('dev').lower()
            if dev == 'user':
                x = x_offset_user
            elif dev == 'dpu':
                x = x_offset_dpu
            elif dev == 'cpu':
                x = x_offset_cpu
            else:
                continue

            y = kernel.get('id') * y_offset

            nodes.append({
                'name': "SubGraph-%d" % kernel.get('id'),
                'title': kernel.get('name'),
                'category': dev,
                'x': x,
                'y': y,
                'input_tensor': [],
                'output_tensor': []
            })

            self.nodes = nodes

        return head + nodes


def makeTreeGridTableNoGraph(fineGrained=False):
    treeGridTable = []
    subGraphStatTable = {}

    for sub in subGraphStat:
        subgTime = subGraphStat[sub]
        minTus = min(subgTime) * 1000 * 1000
        maxTus = max(subgTime) * 1000 * 1000
        meanTus = sum(subgTime) / len(subgTime) * 1000 * 1000
        subGraphStatTable.update({sub: [minTus, maxTus, meanTus]})

        kernel = {
            'title': sub,
            'key': sub,
            'workload': 0,
            'time': meanTus,
            'timeMin': minTus,
            'timeMax': maxTus,
            'icon': False,
        }

        treeGridTable.append(kernel)

    return treeGridTable


class xirParser(parser.parserBase.Parser):
    def __init__(self):
        super().__init__('XIR')

    def parse(self, data, options):
        data = []

        runmode = options.get('runmode', "")
        if runmode == "debug":
            fineGrained = True
        else:
            fineGrained = False

        """Can Only support 1 .xmodel"""
        if len(data) == 0:
            graph = None
        else:
            graph = data[0]['graph']

        if graph is not None:
            t = xirGraph(fineGrained)
            t.addGraph(graph)

            links = t.makeLinks()
            nodes = t.makeNodes()
            "Tree Grid Table"
            tgt = t.makeTreeGridTable()
        else:
            links = []
            nodes = []
            tgt = makeTreeGridTableNoGraph(fineGrained)
        return {'xir-links': links, 'xir-nodes': nodes, 'xir-tgt': tgt}


parser.parserBase.register(xirParser())
