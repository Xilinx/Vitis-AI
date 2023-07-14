"""
 Copyright 2019 Xilinx Inc.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import time
from collections import OrderedDict
from typing import Any, Dict, List, NoReturn, Optional

import numpy as np
import torch
from torch.onnx.utils import OperatorExportTypes
import graphviz

methods_OP = [
    "attributeNames",
    "hasMultipleOutputs",
    "hasUses",
    "inputs",
    "kind",
    "outputs",
    "outputsSize",
    "scopeName",
]
# Some additional methods to explure for methods_IO are
#
#   'unique' (type int)
#   'type' (type <Tensor<class 'torch._C.Type'>>)
#
# But the below are sufficient for now.
methods_IO = ["node", "offset", "debugName", "unique", "type"]


class NodePy(object):
    def __init__(self, node_cpp):
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.kind: str = ""
        self.scope: str = ""

    def __repr__(self):
        repr = []
        repr.append(str(type(self)))
        for m in dir(self):
            if "__" not in m:
                repr.append(
                    m + ": " + str(getattr(self, m)) + str(type(getattr(self, m)))
                )
        return "\n".join(repr) + "\n\n"


class NodePyIO(NodePy):
    def __init__(self, node_cpp, input_or_output=None):
        super(NodePyIO, self).__init__(node_cpp)

        self.offset: int = 0
        self.debug_name: str = ""
        self.id: int = -1
        self.input_or_output: str = ""
        self.tensor_shape: List[int] = None
        self.tensor: np.ndarray = None

        if node_cpp is not None:
            # offset
            self.offset = node_cpp.offset()  # getattr(node_cpp, 'offset')()
            # debugName
            self.debug_name = node_cpp.debugName()  # getattr(node_cpp, 'debugName')()
            # unique
            self.id = node_cpp.unique()  # getattr(node_cpp, 'unique')()
            # kind
            self.kind = "Parameter"
            if input_or_output:
                self.input_or_output = input_or_output
                self.kind = "IO Node"
            # type
            try:
                tensor_size = node_cpp.type().sizes()
            except RuntimeError:
                tensor_size = [
                    1,
                ]  # fail when constant model is used.
            self.tensor_shape = tensor_size


class NodePyOP(NodePy):
    def __init__(self, node_cpp):
        super(NodePyOP, self).__init__(node_cpp)

        self.attributes: Dict[str, Any] = {}
        # self.attributes = ""
        self.has_multiple_outputs: bool = False
        self.has_users: bool = False
        self.outputs_size: int = 0
        self.inputs_tensor_shape: List[List[int]] = []
        self.outputs_tensor_shape: List[List[int]] = []
        self.alias: str = ""
        self.op_name: str = ""
        self.op_type: str = ""

        if node_cpp is not None:
            # attributeNames
            # Replace single quote which causes strange behavior in TensorBoard
            for attr_name in node_cpp.attributeNames():
                assert (
                    attr_name not in self.attributes
                ), f"[ERROR] duplicated attribute name: {attr_name}"
                self.attributes[attr_name] = node_cpp[attr_name]
            # self.attributes = str(
            #     {k: node_cpp[k] for k in node_cpp.attributeNames()}
            # ).replace("'", " ")
            # hasMultipleOutputs
            self.has_multiple_outputs = node_cpp.hasMultipleOutputs()
            # hasUses
            self.has_users = node_cpp.hasUses()
            # kind
            self.kind = node_cpp.kind()
            # scopeName
            self.scope = node_cpp.scopeName()

            # outputsSize
            self.outputs_size = node_cpp.outputsSize()

            for m in ["inputs", "outputs"]:
                list_of_node = list(getattr(node_cpp, m)())
                io_debug_names = []
                io_tensor_shapes = []
                for n in list_of_node:
                    io_debug_names.append(n.debugName())
                    if n.type().kind() == "TensorType":
                        io_tensor_shapes.append(n.type().sizes())
                    else:
                        io_tensor_shapes.append(None)
                setattr(self, m, io_debug_names)
                setattr(self, m + "_tensor_shape", io_tensor_shapes)


class GraphPy(object):
    def __init__(self, root_scope_name="default"):
        self.nodes_op = []
        self.nodes_io = OrderedDict()
        self.debug_name_to_scoped_name = {}
        self.shallowest_scope_name = root_scope_name
        self.scope_name_appeared = []
        self.alias_to_scope = {}

    def append(self, x):
        if isinstance(x, NodePyIO):  # append NodePyIO
            self.nodes_io[x.debug_name] = x

        elif isinstance(x, NodePyOP):  # append NodePyOP
            self.nodes_op.append(x)
            # deal with outputs
            for output_debug_name, output_shape in zip(
                x.outputs, x.outputs_tensor_shape
            ):
                # self.scope_name_appeared.append(x.scope)
                node_io = NodePyIO(None)
                node_io.debug_name = output_debug_name
                node_io.scope = x.scope
                node_io.kind = x.kind
                node_io.inputs = x.inputs
                node_io.outputs = x.outputs
                self.nodes_io[output_debug_name] = node_io

        else:
            raise TypeError(f"[ERROR] Unsupported node type: {x.__class__.__name__}")

    def printall(self):
        print("all nodes")
        for node in self.nodes_op:
            print(node)
        for debug_name in self.nodes_io:
            print(self.nodes_io[debug_name])

    def find_common_root(self):
        for fullscope in self.scope_name_appeared:
            if fullscope:
                self.shallowest_scope_name = fullscope.split("/")[0]

    def populate_namespace_from_OP_to_IO(self):
        for node in self.nodes_op:
            for inode_debug_name in node.inputs:
                self.debug_name_to_scoped_name[inode_debug_name] = (
                    node.scope + "/" + inode_debug_name
                )

        for debug_name, node in self.nodes_io.items():
            if hasattr(node, "input_or_output") and node.input_or_output in [
                "input",
                "output",
            ]:  # input or output nodes
                if debug_name in self.debug_name_to_scoped_name:
                    continue

                self.debug_name_to_scoped_name[debug_name] = (
                    node.input_or_output + "/" + node.debug_name
                )
            elif hasattr(node, "scope"):
                # self.debug_name_to_scoped_name[debug_name] = node.scope + '/' + node.uniqueName
                if node.scope == "" and self.shallowest_scope_name:
                    self.debug_name_to_scoped_name[debug_name] = (
                        self.shallowest_scope_name + "/" + debug_name
                    )
                else:
                    self.debug_name_to_scoped_name[debug_name] = (
                        node.scope + "/" + debug_name
                    )

        # replace debug name in 'inputs' with scope name
        for debug_name, node in self.nodes_io.items():
            if debug_name in self.debug_name_to_scoped_name:
                node.scope = self.debug_name_to_scoped_name[debug_name]

    # def build_alias_to_scope_dict(self):
    #     assert self.nodes_op is not None and len(self.nodes_op) > 0
    #     for node_op in self.nodes_op:
    #         assert (
    #             node_op.alias not in self.alias_to_scope
    #         ), f"[ERROR] Duplicated alias: {node_op.alias}"
    #         self.alias_to_scope[node_op.alias] = node_op.scope


def parse_model(
    model: torch.nn.Module,
    inputs_shape: List[List[int]],
    ignore_useless_nodes: bool = True,
    dump_image: bool = False,
) -> GraphPy:
    assert model is not None, "'model' should not be None."
    assert inputs_shape is not None, "'input_shape' should not be None."

    dummy_inputs = []
    for shape in inputs_shape:
        dummy_inputs.append(torch.randn(shape))
    model.eval()

    trace, _ = torch.jit.get_trace_graph(model, tuple(dummy_inputs))
    optimize_trace(trace)
    graph = trace.graph()

    # number of input nodes
    n_inputs = len(dummy_inputs)

    state_dict = torch.jit._unique_state_dict(model)
    # state_names: List[str] = list(state_dict.keys())
    state_values = list(state_dict.values())

    graph_py = GraphPy(root_scope_name=model.__class__.__name__)
    # NodePyIO for inputs and parameters
    for i, node in enumerate(graph.inputs()):
        if ignore_useless_nodes:
            if (
                len(node.uses()) == 0
            ):  # number of user of the node (= number of outputs/ fanout)
                continue

        if i < n_inputs:  # the first n nodes are input nodes
            node_io = NodePyIO(node, "input")
        else:
            node_io = NodePyIO(node)
            node_io.tensor = state_values[i - n_inputs]
        graph_py.append(node_io)  # parameter

    # NodePyOP for cases except for iputs, parameters, and outputs
    for node in graph.nodes():
        graph_py.append(NodePyOP(node))

    # NodePyIO for outputs
    for node in graph.outputs():  # must place last.
        graph_py.append(NodePyIO(node, "output"))

    graph_py.find_common_root()
    graph_py.populate_namespace_from_OP_to_IO()

    # update the 'outputs' fields
    for debug_name, node in graph_py.nodes_io.items():
        if len(node.inputs) > 0:
            for iname in node.inputs:
                inode = graph_py.nodes_io.get(iname)
                assert inode is not None, "[ERROR] Not found input node: name {iname}"
                inode.outputs.append(debug_name)

    def extract_alias(scope: str) -> str:
        if "[" not in scope:
            return ""
        res = []
        start = -1
        for i, ch in enumerate(scope):
            if ch == "[":
                start = i + 1
            elif ch == "]":
                res.append(scope[start:i])
                start = -1
        return ".".join(res)

    # * extract inputs and outputs
    input_dict: Dict[str, NodePyIO] = {}
    output_dict: Dict[str, NodePyIO] = {}
    keys: List[str] = graph_py.nodes_io.keys()
    for key in keys:
        node = graph_py.nodes_io[key]
        if node.input_or_output == "input":
            input_dict[key] = node
        elif node.input_or_output == "output":
            output_dict[key] = node

    # * extract ops
    op_dict = {}
    for node in graph_py.nodes_op:
        node.alias = node.outputs.pop(0)

        # set the 'op_type' field
        node.op_type = node.kind.split("::")[-1].replace("_", "")

        # set the 'op_name' field
        node.op_name = extract_alias(node.scope)

        # add current node to op_dict
        assert node.alias not in op_dict
        op_dict[node.alias] = node

    # * update op type of nodes_op in graph_py
    revise_op_type(graph_py.nodes_op)

    # * update scope name
    revise_scope(graph_py.nodes_op)

    # * update the 'outputs' field of the input nodes
    for _, inode in input_dict.items():
        inode.outputs = [x for x in inode.outputs if x in op_dict]

    # * update the 'outputs' field of each node_op
    for node in graph_py.nodes_op:
        if len(node.inputs) > 0:
            for iname in node.inputs:
                inode = op_dict.get(iname)
                if inode is not None:
                    if node.alias not in inode.outputs:
                        inode.outputs.append(node.alias)
                else:
                    if iname not in graph_py.nodes_io:
                        raise KeyError(f"input name: {iname}")

    # ! debug: set op name
    update_op_name(graph_py, op_dict)

    # * dump graph
    if dump_image:
        dump(op_dict)

    return graph_py


def optimize_trace(trace):
    trace.set_graph(optimize_graph(trace.graph()))


def optimize_graph(graph):
    # we record some ops like ones/zeros
    # into a trace where we previously recorded constants
    # use constant prop to maintain our current level of onnx support
    # without implementing symbolics for all of them
    torch._C._jit_pass_constant_propagation(graph)
    torch.onnx.utils._split_tensor_list_constants(graph, graph)
    # run dce to eliminate dead parts of the graph that might have been
    # left behind by things like symbolic_override
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)

    # torch._C._jit_pass_canonicalize_ops(graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_peephole(graph, True)
    torch._C._jit_pass_lint(graph)

    # onnx only supports tensors, but 1 / 2 = 0.5 and tensor(1) / tensor(2) = 0
    torch._C._jit_pass_prepare_division_for_onnx(graph)
    # onnx only supports tensors, so we turn all out number types into tensors
    torch._C._jit_pass_erase_number_types(graph)
    # onnx does not support tuples, so try to remove them
    torch._C._jit_pass_lower_all_tuples(graph)
    torch._C._jit_pass_peephole(graph, True)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_fixup_onnx_loops(graph)
    torch._C._jit_pass_lint(graph)
    graph = torch._C._jit_pass_canonicalize(graph)
    torch._C._jit_pass_lint(graph)
    return graph


def dump(op_dict, filename="test") -> NoReturn:
    # create a DG instance with svg format
    graph = graphviz.Digraph(format="svg")

    # create nodes for graph
    for op_name, node in op_dict.items():
        graph.node(op_name, label=node.op_type)

    # create edges for graph
    for op_name, node in op_dict.items():
        if len(node.outputs) > 0:
            for cname in node.outputs:
                graph.edge(op_name, cname)

    # render and save the graph
    graph.render(filename, directory=None, view=False, cleanup=False)


def revise_op_type(nodes_op) -> NoReturn:
    assert nodes_op is not None, "'nodes_op' should not be None."
    for node in nodes_op:
        assert node.op_type is not None and len(node.op_type) > 0
        if node.op_type == "PythonOp" and node.kind.startswith("prim::"):
            if "PythonOp" in node.kind:
                # extract class name from the scope field and use it as the op type
                class_name = node.scope.split("/")[-1]
                if "[" in class_name:
                    class_name = class_name.split("[")[0]
                node.op_type = class_name.lower()


def revise_scope(nodes_op) -> NoReturn:
    assert nodes_op is not None, "'nodes_op' should not be None."

    seq_counter = 1
    blk_counter = 0
    for node_op in nodes_op:
        scope_name = node_op.scope
        if "DetNet/Sequential" in scope_name:
            names = [x for x in scope_name.split("/")]
            assert "Quant_BasicBlock" in names[2]
            start = names[2].index("[") + 1
            end = names[2].index("]")
            assert end > start
            curr_blk_counter = int(names[2][start:end])
            if blk_counter == curr_blk_counter:
                names[1] = names[1] + "[layer" + str(seq_counter) + "]"
            elif blk_counter + 1 == curr_blk_counter:
                blk_counter = curr_blk_counter
                names[1] = names[1] + "[layer" + str(seq_counter) + "]"
            elif blk_counter > curr_blk_counter:
                blk_counter = curr_blk_counter
                seq_counter += 1
                names[1] = names[1] + "[layer" + str(seq_counter) + "]"
            else:
                raise ValueError(
                    f"[ERROR] blk_counter: {blk_counter}, curr_blk_counter: {curr_blk_counter}"
                )

            node_op.scope = "/".join(names)


def update_op_name(graph_py, op_dict):
    assert graph_py is not None, "'graph_py' should not be None."
    assert op_dict is not None, "'op_dict' should not be None."

    # update op name
    for node in graph_py.nodes_op:
        node.op_name = node.scope + "/" + node.alias

    # update inputs and outputs of node_op
    for node in graph_py.nodes_op:
        if node.inputs is not None and len(node.inputs) > 0:
            for i, iname in enumerate(node.inputs):
                if iname in op_dict:
                    inode = op_dict.get(iname)
                    # update iname
                    node.inputs[i] = inode.scope + "/" + inode.alias
        if node.outputs is not None and len(node.outputs) > 0:
            for i, oname in enumerate(node.outputs):
                if oname in op_dict:
                    onode = op_dict.get(oname)
                    # update oname
                    node.outputs[i] = onode.scope + "/" + onode.alias

    # update inputs and outputs of node_io
    for _, node_io in graph_py.nodes_io.items():
        # update inputs
        if node_io.inputs is not None and len(node_io.inputs) > 0:
            for i, iname in enumerate(node_io.inputs):
                if iname in op_dict:
                    inode = op_dict.get(iname)
                    # update iname
                    node_io.inputs[i] = inode.op_name
        # update outputs
        if node_io.outputs is not None and len(node_io.outputs) > 0:
            for i, oname in enumerate(node_io.outputs):
                if oname in op_dict:
                    onode = op_dict.get(oname)
                    # update oname
                    node_io.outputs[i] = onode.op_name
