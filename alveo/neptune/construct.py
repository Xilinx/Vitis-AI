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
"""
This package constructs services from the JSON representation of the recipe.

If the service uses any artifacts, those are downloaded. The service graph is parsed,
the appropriate nodes are imported and the created service is added to the
ServiceManager. For Tornado, the handlers are assigned and the Tornado URL
is assigned. It returns a tuple that can be added to Tornado.
"""


import importlib
import os
from zipfile import ZipFile

import tornado.web
import wget

import neptune.service as service
from neptune import node_manager
from neptune.common import MultiDimensionalArrayEncoder
from neptune.service_manager import ServiceManager
from vai.dpuv1.rt.xsnodes import grapher


def get_artifacts(args):
    for arg in args:
        if 'artifact' in arg:
            zf = os.path.join(arg['artifact']['path'],arg['artifact']['key']+'.zip')
            if not os.path.exists(zf):
                wget.download(arg['artifact']['url'],zf)
                print (" ")
                with ZipFile(zf,'r') as fp:
                    fp.extractall(arg['artifact']['path'])
                print (" ")

def parse_graph(graph, args, name):
    manager = node_manager.NodeManager()
    graphObj = grapher.Graph(name)

    use_const_args = True if len(args) == 1 else False

    service_args = []
    for index, node in enumerate(graph['nodes']):
        node_path = manager.get_path(node)
        try:
            module = importlib.import_module(str(node_path))
        except Exception as e:
            print(e)
        node_args = args[0] if use_const_args else args[index]
        graphObj.node('n_' + str(index), module.Node, node_args, node)

        try:
            node_args = module.Args().to_dict()
            node_args['name'] = node
            service_args.append(node_args)
        except AttributeError:
            service_args.append(None)

    for edge in graph['edges']:
        (start, end, index) = edge
        if start == -1:
            graphObj.edge("e" + str(index), None, "n_" + str(end))
        elif end == -1:
            graphObj.edge("e" + str(index), "n_" + str(start), None)
        else:
            graphObj.edge("e" + str(index), "n_" + str(start), "n_" + str(end))

    return graphObj, module, service_args

def generate_hash(graph):
    hash_name = ""
    for node in graph['nodes']:
        hash_name += node + "_"

    for edge in graph['edges']:
        (start, end) = edge
        hash_name += str(start) + str(end)

    return hash_name

class GenericHandler(tornado.web.RequestHandler):
    def initialize(self, handlers):
        self.handlers = handlers

    async def get(self):
        if 'get' in self.handlers:
            get_handlers = self.handlers['get']

            result = await get_handlers[0](self)
            for handler in get_handlers[1:]:
                result = await handler(self, result)
            return result

    async def post(self):
        if 'post' in self.handlers:
            post_handlers = self.handlers['post']

            result = await post_handlers[0](self)
            for handler in post_handlers[1:]:
                result = await handler(self, result)
            return result

    async def patch(self):
        if 'patch' in self.handlers:
            patch_handlers = self.handlers['patch']

            result = await patch_handlers[0](self)
            for handler in patch_handlers[1:]:
                result = await handler(self, result)
            return result

def _construct(graph, args, name, url, handlers):
    # Plasma, for whatever reason, enforces this for their IDs. Since these
    # names are used as IDs, we have to enforce this too. This name gets
    # appended with other strings (the overall length must be <20 chars) so
    # ours needs to be less than 12 chars
    if len(name) >= 12:
        print("Cannot construct service, name too long (must be <12 chars")
        return

    get_artifacts(args)

    (graphObj, module, args) = parse_graph(graph, args, name)

    if url == '':
        url = generate_hash(graph)
    serve_url = "/serve/%s" % url

    new_service = service.Service(name, {}, graphObj)
    ServiceManager().add(name, new_service, serve_url, args)

    handler_methods = {'service': name}
    if 'get' in handlers:
        get_methods = []
        for method in handlers['get']:
            handler_module = 'neptune.handlers.' + method.split('.')[0]
            module_0 = __import__(handler_module, fromlist=[method.split('.')[1]])
            func = getattr(module_0, method.split('.')[1])
            get_methods.append(func)
        handler_methods['get'] = get_methods
    if 'post' in handlers:
        post_methods = []
        for method in handlers['post']:
            handler_module = 'neptune.handlers.' + method.split('.')[0]
            module_0 = __import__(handler_module, fromlist=[method.split('.')[1]])
            func = getattr(module_0, method.split('.')[1])
            post_methods.append(func)
        handler_methods['post'] = post_methods
    if 'patch' in handlers:
        patch_methods = []
        for method in handlers['patch']:
            handler_module = 'neptune.handlers.' + method.split('.')[0]
            module_0 = __import__(handler_module, fromlist=[method.split('.')[1]])
            func = getattr(module_0, method.split('.')[1])
            patch_methods.append(func)
        handler_methods['patch'] = patch_methods

    return [(r"" + serve_url, GenericHandler, dict(handlers=handler_methods))]

def construct(dict_in):
    name = str(dict_in['name'])
    recipe_cache = os.environ["VAI_ALVEO_ROOT"] + "/neptune/recipes/recipe_%s.bak" % name
    enc = MultiDimensionalArrayEncoder()
    with open(recipe_cache, 'w') as f:
        f.write(enc.encode(dict_in))
    return _construct(
        dict_in['graph'],
        dict_in['args'],
        name,
        str(dict_in['url']),
        dict_in['handlers']
    )
