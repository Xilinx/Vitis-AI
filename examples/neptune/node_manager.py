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
from neptune.common import list_submodules, remove_prefix

class NodeManager(object):
    class __impl:
        """ Implementation of the singleton interface """

        def __init__(self):
            self._prefix = 'vai.dpuv1.rt.xsnodes'
            self._node_paths = list_submodules(self._prefix)
            self._node_names = [remove_prefix(s, self._prefix + '.') for s in self._node_paths]

        def check_id(self):
            """ Test method, return singleton id """
            return id(self)

        def get_paths(self):
            return self._node_paths

        def get_names(self):
            return self._node_names

        def get_path(self, name):
            for index, node in enumerate(self._node_names):
                if name == node:
                    return self._node_paths[index]
            print("Node %s not found!" % name)
            return None

    # storage for the instance reference
    __instance = None

    def __init__(self):
        """ Create singleton instance """
        # Check whether we already have an instance
        if NodeManager.__instance is None:
            # Create and remember instance
            NodeManager.__instance = NodeManager.__impl()

        # Store instance reference as the only member in the handle
        self.__dict__['_Node_Manager__instance'] = NodeManager.__instance

    def __getattr__(self, attr):
        """ Delegate access to implementation """
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """ Delegate access to implementation """
        return setattr(self.__instance, attr, value)

    def _drop(self):
        "Drop the instance (for testing purposes)."
        NodeManager.__instance = None
        del self._Node_Manager__instance
