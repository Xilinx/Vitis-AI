

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

import operator
from .search import breadth_first_search_handler,depth_first_search_handler
#from xir import Graph
from xir import Graph
from xir import Op
from xir import Tensor
from utils.tools import *

def dfs_sort_xir_graph_nodes(graph, nodes):
    MaxDepth = {}
    cur_depth = [0]
    
    def __set_max_depth(node_name):
        MaxDepth[node_name] = cur_depth[0]
        cur_depth[0] += 1
    
    def __children_names(node_name):
        if node_name == 'XirFooStart':
            for n in nodes:
                if n.get_input_num()==0 and 'bias' in n.get_name():
                    yield n.get_name()
        else:
            find_one_deep_op = 0
            for c in sorted(__children_nodes(nodes, node_name), key=lambda c:c.get_name(), reverse=True):
                if all(p_node.get_name() in MaxDepth for p_node in get_input_ops_list(c)):
                    find_one_deep_op = 1
                    yield c.get_name()
                else:
                    continue
            if 0 == find_one_deep_op:
                for c in sorted(__children_nodes(nodes,node_name), key=lambda c:c.get_name(), reverse=True):
                    combine_ops = []
                    data_ops = []
                    weights_ops = []
                    bias_ops = []
                    for p_node in get_input_ops_list(c):
                        if p_node.get_input_num()>0 and p_node.get_name() not in MaxDepth:
                            combine_ops.append(p_node.get_name())
                        elif p_node.get_name() not in MaxDepth and p_node.get_type()=='data':
                            data_ops.append(p_node.get_name())
                        elif p_node.get_name() not in MaxDepth and p_node.get_type()=='const' and  \
                            'weights' in p_node.get_name():
                            weights_ops.append(p_node.get_name())
                        elif p_node.get_name() not in MaxDepth and p_node.get_type()=='const' and  \
                            'bias' in p_node.get_name():
                            bias_ops.append(p_node.get_name())
                    
                    for node in combine_ops:
                        yield node
                    for node in data_ops:
                        yield node
                    for node in weights_ops:
                        yield node
                    for node in bias_ops:
                        yield node
                    '''
                    for p_node in get_input_ops_list(c) if p_node.get_input_num()==0 and  \
                        p_node.get_name() not in MaxDepth and p_node.get_type()=='data':
                        yield p.node.get_name()
                        
                    for p_node in get_input_ops_list(c) if p_node.get_input_num()==0 and  \
                        p_node.get_name() not in MaxDepth and p_node.get_type()=='const' and  \
                        'weights' in p_node.get_name():
                        yield p.node.get_name()
                        
                    for p_node in get_input_ops_list(c) if p_node.get_input_num()==0 and  \
                        p_node.get_name() not in MaxDepth and p_node.get_type()=='const' and  \
                        'bias' in p_node.get_name():
                        yield p.node.get_name()
                    '''
                        
    def __children_nodes(nodes, name):
        children_nodes = []
        for n in nodes:
            if n.get_input_num() > 0 and name in [op.get_name() for op in get_input_ops_list(n)]:
                children_nodes.append(n)
        return children_nodes
    
    depth_first_search_handler('XirFooStart', generator=__children_names, handler=__set_max_depth)
    assert len(MaxDepth)-1==max(MaxDepth.values()),"max depth mismatch, has {} nodes and max depth of {}, please check!".format(len(MaxDepth)-1,max(MaxDepth.values()))
    #print(MaxDepth)
    sorted_nodes=sorted(nodes, key=lambda n:MaxDepth[n.get_name()])
    return sorted_nodes, MaxDepth

def dfs_sort_xir_graph_nodes_reverse(graph, nodes):
    MaxDepth = {}
    cur_depth = [0]
    
    def __set_max_depth(node_name):
        MaxDepth[node_name] = cur_depth[0]
        cur_depth[0] += 1
    
    def __children_names(node_name):
        if node_name == 'XirFooStart':
            for n in nodes:
                if len(__children_nodes(nodes, n.get_name()))==0:
                    yield n.get_name()
        else:
            find_one_deep_op = 0
            for c in sorted(__parent_nodes(node_name), key=lambda c:c.get_name(), reverse=True):
                if all(p_node.get_name() in MaxDepth for p_node in __children_nodes(nodes, c.get_name())):
                    find_one_deep_op = 1
                    yield c.get_name()
                else:
                    continue
                        
    def __children_nodes(nodes, name):
        children_nodes = []
        for n in nodes:
            if n.get_input_num() > 0 and name in [op.get_name() for op in get_input_ops_list(n)]:
                children_nodes.append(n)
        return children_nodes
    
    def __parent_nodes(name):
        node = graph.get_op(name)
        if node.get_input_num() == 0:
            return []
        else:
            return get_input_ops_list(node)
    
    depth_first_search_handler('XirFooStart', generator=__children_names, handler=__set_max_depth)
    assert len(MaxDepth)-1==max(MaxDepth.values()),"max depth mismatch, has {} nodes and max depth of {}, please check!".format(len(MaxDepth)-1,max(MaxDepth.values()))
    #print(MaxDepth)
    sorted_nodes=sorted(nodes, key=lambda n:MaxDepth[n.get_name()], reverse=True)
    return sorted_nodes, MaxDepth

def reorder_xir_graph(graph, xir_sort_strategy='dfs'):
    #print('This is reorder xir graph function')
    
    xir_nodes = graph.toposort()
    
    '''
    input0_node = None
    input1_node = None
    for node in xir_nodes:
        if node.get_type() == 'data' and 'input_0' in node.get_name():
            input0_node = node
            input0_index = xir_nodes.index(node)
        elif node.get_type() == 'data' and 'input_1' in node.get_name():
            input1_node = node
            input1_index = xir_nodes.index(node)
    
    print(input0_node.get_name(), input0_index)
    print(input1_node.get_name(), input1_index)
    
    xir_nodes[input0_index], xir_nodes[input1_index] = xir_nodes[input1_index], xir_nodes[input0_index]
    print([node.get_name() for node in xir_nodes])
    '''
    
    if xir_sort_strategy == 'bfs':
        sorted_nodes = xir_nodes
        #sorted_nodes = bfs_sort_xir_gragh_nodes(graph)
    elif xir_sort_strategy == 'dfs':
        sorted_nodes, _ = dfs_sort_xir_graph_nodes_reverse(graph, xir_nodes)
    else:
        raise KeyError('xir_sort_strategy {} is not supported'.format(xir_sort_strategy))
    
    return sorted_nodes
