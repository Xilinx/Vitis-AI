

#
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
#

from .utils import FIFOQueue

#depth first search, return all end_nodes that fit to critierion
def get_dfs_end_nodes(start_nodes,critierion,generator):
    nodes_map={}
    for s_node in start_nodes:
        if s_node not in nodes_map:
            frontier=[s_node]
            explored=set()
            nodes_map[s_node]=[]
            while frontier:
                node=frontier.pop()
                if critierion(node):
                    nodes_map[s_node].append(node)
                else:
                    explored.add(node)
                    frontier.extend([n for n in generator(node) if n not in explored and n not in frontier])
    return nodes_map

def graph_search_handler(start_node,generator,frontier,handler=None,gen_params={}):
    frontier.append(start_node)
    explored = set()
    while frontier:
        node = frontier.pop()
        explored.add(node)
        if handler:
            handler(node)
        frontier.extend([n for n in generator(node,**gen_params) if n not in explored and n not in frontier])
    return None

def breadth_first_search_handler(start_node,generator,handler=None,gen_params={}):
    return graph_search_handler(start_node,generator,frontier=FIFOQueue(),handler=handler,gen_params=gen_params)

def depth_first_search_handler(start_node,generator,handler=None,gen_params={}):
    return graph_search_handler(start_node,generator,frontier=[],handler=handler,gen_params=gen_params)
