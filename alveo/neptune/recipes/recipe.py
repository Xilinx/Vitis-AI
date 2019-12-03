# Copyright 2019 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json

from neptune.common import MultiDimensionalArrayEncoder

class Construct(object):
    def __init__(self, graph=None, name=None, url=None, handlers=None):
        self.graph = graph
        self.name = name
        self.url = url
        self.handlers = handlers
        self._encoder = MultiDimensionalArrayEncoder()
        self._finished = False

    def _to_dict(self):
        if not self._finished:
            self.graph._finish()
            self._finished = True
        retval = {}
        retval['graph'] = self.graph.to_dict()
        retval['args'] = self.graph.args
        retval['name'] = self.name
        retval['url'] = self.url
        retval['handlers'] = self.handlers.to_dict()
        return retval

    def to_json(self):
        retval = self._to_dict()
        return json.loads(self._encoder.encode(retval))

    def to_dict(self):
        return self._to_dict()

class Graph(object):
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.args = []
        self._nodes = {}
        self.START = -1
        self.END = -1

    def add_node(self, node, args={}):
        assert isinstance(node, str)
        assert isinstance(args, dict)

        self._nodes[node] = len(self.nodes)
        self.nodes.append(node)
        self.args.append(args)

    def _get_index(self, start):
        index = len(self.edges) + 1
        for edge in self.edges:
            _start, _end, _index = edge
            if self._nodes[start] == _start:
                index = _index
        return index

    def insert_edge(self, position, start, end):
        index = self._get_index(start)
        self.edges.insert(position, (self._nodes[start], self._nodes[end], index))

    def add_edge(self, start, end):
        index = self._get_index(start)
        self.edges.append((self._nodes[start], self._nodes[end], index))

    def set_arg(self, node, key, value):
        self.args[self._nodes[node]][key] = value

    def _finish(self):
        self.edges.insert(0, (self.START, 0, 0))
        start, _end, _index = self.edges[-1]
        if start == len(self.nodes) - 1:
            index = _index
        else:
            index = _index + 1
        self.edges.append((len(self.nodes) - 1, self.END, index))
        del self._nodes

    def to_dict(self):
        retval = {}
        retval['nodes'] = self.nodes
        retval['edges'] = self.edges
        return retval

class Handlers(object):
    def __init__(self, get=None, post=None, patch=None):
        if get is None:
            self.get = []
        else:
            if isinstance(get, str):
                self.get = [get]
            elif isinstance(get, list):
                self.get = get
            else:
                self.get = []
        if post is None:
            self.post = []
        else:
            if isinstance(post, str):
                self.post = [post]
            elif isinstance(post, list):
                self.post = post
            else:
                self.post = []
        if patch is None:
            self.patch = []
        else:
            if isinstance(patch, str):
                self.patch = [patch]
            elif isinstance(patch, list):
                self.patch = patch
            else:
                self.patch = []

    def add_get_handler(self, get):
        self.get.append(get)

    def add_post_handler(self, post):
        self.post.append(post)

    def add_patch_handler(self, patch):
        self.patch.append(patch)

    def to_dict(self):
        retval = {}
        if self.get:
            retval['get'] = self.get
        if self.post:
            retval['post'] = self.post
        if self.patch:
            retval['patch'] = self.patch
        return retval
