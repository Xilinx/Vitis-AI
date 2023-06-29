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

import logging
import sys
from pathlib import Path
from typing import Dict, List, NoReturn, Optional, Tuple

import graphviz
from tqdm import tqdm
from xnnc.ir.enums import *
from xnnc.ir.xnode import (
    REGISTERED_OPS,
    LayoutType,
    XModelNode,
    XModelNodeConst,
    XModelNodeInput,
)
from xnnc.proto.openir import openir
from xnnc.tensor.xtensor import DataFormat

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create console handler and set level to DEBUG
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter
formatter = logging.Formatter("%(name)s - %(lineno)d - %(levelname)s - %(message)s")
# add formatter to ch
ch.setFormatter(formatter)
# add ch to logger
logger.addHandler(ch)


class XModel(object):
    """
    XModel protocol
    """

    def __init__(self, name: str, model_type: str, layout: str = "NCHW"):
        """
        Paramters:
            name: the name of the resulted XModel instance.
            model_type: the type of the orginal model architecture, which must be one of caffe, tensorflow and onnx.
            layout: data format of tensors, which is 'NCHW' (default) or 'NHWC'.
        """
        if name is None or type(name) is not str or len(name) == 0:
            logger.error("Invalid argument: 'name'.")
            sys.exit(1)
        if model_type is None or type(model_type) is not str or len(model_type) == 0:
            logger.error("Invalid argument: 'model_type'.")
            sys.exit(1)
        model_t = model_type.lower()
        if model_t not in [
            "caffe",
            "tensorflow",
            "pytorch",
            "onnx",
            "tensorflow2",
            "nndct",
        ]:
            err_msg = f"[ERROR] Unsupported model type: {model_t}. The expected value is 'caffe', 'tensorflow', 'tensorflow2', 'nndct'."
            logger.error(err_msg)
            print(err_msg)
            sys.exit(1)
        if layout is None or type(layout) is not str or len(layout) == 0:
            logger.error("Invalid argument: 'layout'.")
            sys.exit(1)
        model_l = layout.upper()
        if model_l not in ["NCHW", "NHWC"]:
            logger.error(
                "The value of argument 'layout' shoud be 'NCHW'(default) or 'NHWC'"
            )
            sys.exit(1)

        self.__name: str = name
        # the original model: caffe, tensorflow, pytorch, etc.
        self.__origin: str = model_t
        self.__layout: str = model_l
        self.__input_xnodes: List[XModelNode] = []
        self.__output_xnodes: List[XModelNode] = []
        self.__is_quantized: bool = False
        # True: the model has done shape inference; otherwise, False
        self.__shape_inferred: bool = False

        # internal dict
        self.__xnode_dict: Dict[str, XModelNode] = {}

        # ! experimental
        self.flatten_stack = []

    @property
    def name(self) -> str:
        """
        Get the name of the XModel object.

        Returns
        -------
        str
            name of the XModel object.
        """
        return self.__name

    @property
    def origin(self) -> str:
        """
        Get the origin model type.

        Returns
        -------
        str
            origin model type.
        """
        return self.__origin

    @property
    def layout(self) -> str:
        """
        Get the layout of the XModel object.

        Returns
        -------
        str
            layout info, "NCHW" or "NHWC".
        """
        return self.__layout

    @layout.setter
    def layout(self, layout: str) -> None:
        """
        Set the layout info for the XModel object.

        Parameters
        ----------
        layout : str
            layout info, "NCHW" or "NHWC".
        """
        assert layout is not None, "'layout' should not be None."
        assert isinstance(layout, str), "'layout' should be of str type."
        assert layout in ["NCHW", "NHWC"], "'layout' must be 'NCHW' or 'NHWC'."
        self.__layout = layout

    @property
    def xnodes(self) -> List[XModelNode]:
        """
        Get the XModelNode objects contained in the XModel object.

        Returns
        -------
        List[XModelNode]
            list of XModelNode objects.
        """
        return list(self.__xnode_dict.values())

    @property
    def input(self) -> List[XModelNode]:
        """Get input xnodes, which have zero in-degree.

        Returns
        -------
        List[XModelNode]
            list of XModelNode objects.
        """
        return self.__input_xnodes

    @property
    def output(self) -> List[XModelNode]:
        """Get output xnodes, which have zero out-degree.

        Returns
        -------
        List[XModelNode]
            list of XModelNode objects.
        """
        return self.__output_xnodes

    @property
    def size(self) -> int:
        """
        Return the number of nodes.

        Returns
        -------
        int
            Number of nodes.
        """
        return len(self.__xnode_dict)

    @property
    def is_quantized(self) -> bool:
        """
        Check if the XModel object is a quantized model or not.

        Returns
        -------
        bool
            True, if the XModel object is quantized; otherwise, False.
        """
        return self.__is_quantized

    @is_quantized.setter
    def is_quantized(self, value: bool) -> NoReturn:
        """
        Set the XModel object as a quantized model or not one.

        Parameters
        ----------
        value : bool
            bool value

        Returns
        -------
        NoReturn
            No return value.
        """
        self.__is_quantized = value

    @property
    def shape_inferred(self) -> bool:
        """
        Check if the current XModel object has already performed shape inference or not.

        Returns
        -------
        bool
            True, if the shape inference has already been performed; otherwise, False.
        """
        return self.__shape_inferred

    def add_xnode(self, xnode: XModelNode) -> NoReturn:
        """
        Add an XModelNode object to xmodel. The topological info of the XModelNode object should be maintained before performing the invocation.

        Parameters
        ----------
        xnode : XModelNode
            target XModelNode object to be added.

        Returns
        -------
        NoReturn
            No return value.
        """
        assert xnode is not None, "The argument 'xnode' should not be None."
        assert isinstance(
            xnode, XModelNode
        ), "The argument 'xnode' must be of XModelNode type or its derivations."
        # update xnode dict
        if xnode.op_type not in ['const']:
            assert (
                xnode.op_name not in self.__xnode_dict
            ), f"[ERROR] Trying to insert duplicate node in xmodel: name: {xnode.op_name}, type: {xnode.op_type}."

        # host
        xnode.host = self
        # layout
        if not xnode.layout:
            xnode.layout = self.layout
        if isinstance(xnode, XModelNodeConst):
            xnode.tensor.data_format = DataFormat[xnode.layout]
        self.__xnode_dict[xnode.op_name] = xnode

    def add_xnodes(self, xnodes: List[XModelNode]) -> NoReturn:
        """
        Add a list of XModelNode objects. The topological info of each XModelNode object should be maintained before performing the invocation.

        Parameters
        ----------
        xnodes : List[XModelNode]
            list of XModelNode objects.

        Returns
        -------
        NoReturn
            No return value.
        """
        assert xnodes is not None, "'xnodes' should not be None."
        if len(xnodes) > 0:
            for xnode in xnodes:
                self.add_xnode(xnode)

    def topsort(self) -> NoReturn:
        """
        Topologically sort the nodes.

        Returns
        -------
        NoReturn
            No return value.
        """
        # topsort xnodes
        # * key: name of xnode, value: xnode
        xnode_dict: Dict[str, XModelNode] = {}
        # * key: name of xnode, value: its in-degrees
        indegree_xnode: Dict[str, int] = {}
        # stores xnodes with zero indegree
        zero_nodes = []
        for xnode in self.xnodes:
            xnode_dict[xnode.op_name] = xnode
            indegree_xnode[xnode.op_name] = xnode.indegree

            if len(xnode.bottom) == 0:
                zero_nodes.append(xnode)

        sorted_xnodes = []
        while len(zero_nodes) > 0:
            xnode = zero_nodes.pop(0)
            sorted_xnodes.append(xnode)
            # iterate child xnodes and update their indegrees
            for cname in xnode.top:
                cnode = self.get_xnode_by_name(cname)
                assert cnode is not None
                count = cnode.bottom.count(xnode.op_name)
                if count != 1:
                    print(
                        f"[WARNING] Found non-single incoming edge: expected=1, actual={count}. Op name: {cname}."
                    )
                indegree_xnode[cname] -= count
                # if the indegree of child xnode is zero,
                # put it into zero_xnodes
                if indegree_xnode[cname] == 0:
                    child = xnode_dict.get(cname)
                    assert (
                        child is not None
                    ), f"[ERROR] Not found xnode (name: {cname})."
                    zero_nodes.append(child)

        # reset
        self.__xnode_dict = {}
        # update
        for xnode in sorted_xnodes:
            self.add_xnode(xnode)

    def get_xnode_by_name(self, name: str) -> Optional[XModelNode]:
        """Get an XModelNode instance by name.

        Parameters
        ----------
        name : str
            Name of the XModelNode instance to get back.

        Returns
        -------
        Optional[XModelNode]
            XModelNode instance. If no XModelNode instance matches the name, return None.
        """
        return self.__xnode_dict.get(name, None)

    def remove_xnode(self, xnode: XModelNode) -> bool:
        """
        Remove a specific XModelNode object, and update the topology.

        Parameters
        ----------
        xnode : XModelNode
            target XModelNode object to be removed.

        Returns
        -------
        bool
            True, if the target XModelNode object is removed successfully; otherwise, False.
        """
        if self.has_xnode(xnode.op_name):
            parent_names = xnode.bottom
            child_names = xnode.top
            # update the bottom of parent nodes
            parent_names = xnode.bottom
            if parent_names is not None and len(parent_names) > 0:
                for pname in parent_names:
                    pnode = self.__xnode_dict.get(pname)
                    assert pnode is not None, f"Not found parent node: name: {pname}"
                    if len(pnode.top) > 0:
                        idx = pnode.top.index(xnode.op_name)
                        if child_names is not None and len(child_names) > 0:
                            if idx == 0:
                                pnode.top = (
                                    [x for x in child_names]
                                    if len(pnode.top) == 1
                                    else [x for x in child_names] + pnode.top[1:]
                                )
                            elif idx == len(pnode.top) - 1:
                                pnode.top = pnode.top[:idx] + [x for x in child_names]
                            else:
                                pnode.top = (
                                    pnode.top[:idx]
                                    + [x for x in child_names]
                                    + pnode.top[idx + 1 :]
                                )
                        else:
                            if idx == 0:
                                pnode.top = [] if len(pnode.top) == 1 else pnode.top[1:]
                            elif idx == len(pnode.top) - 1:
                                pnode.top = pnode.top[:idx]
                            else:
                                pnode.top = pnode.top[:idx] + pnode.top[idx + 1 :]

            # update the top of child nodes
            if child_names is not None and len(child_names) > 0:
                for cname in child_names:
                    cnode = self.__xnode_dict.get(cname)
                    assert cnode is not None, f"Not found child node: name: {cname}"
                    if len(cnode.bottom) > 0:
                        idx = cnode.bottom.index(xnode.op_name)
                        if parent_names is not None and len(parent_names) > 0:
                            if idx == 0:
                                cnode.bottom = (
                                    [x for x in parent_names]
                                    if len(cnode.bottom) == 1
                                    else [x for x in parent_names] + cnode.bottom[1:]
                                )
                            elif idx == len(cnode.bottom) - 1:
                                cnode.bottom = cnode.bottom[:idx] + [
                                    x for x in parent_names
                                ]
                            else:
                                cnode.bottom = (
                                    cnode.bottom[:idx]
                                    + [x for x in parent_names]
                                    + cnode.bottom[idx + 1 :]
                                )
                        else:
                            if idx == 0:
                                cnode.bottom = (
                                    [] if len(cnode.bottom) == 1 else cnode.bottom[1:]
                                )
                            elif idx == len(cnode.bottom) - 1:
                                cnode.bottom = cnode.bottom[:idx]
                            else:
                                cnode.bottom = (
                                    cnode.bottom[:idx] + cnode.bottom[idx + 1 :]
                                )

            # remove xnode
            del self.__xnode_dict[xnode.op_name]
            return True
        return False

    def remove_xnodes(self, xnodes: List[XModelNode]) -> bool:
        """
        Remove a list of XModelNode objects.

        Parameters
        ----------
        xnodes : List[XModelNode]
            list of XModelNode objects to be removed.

        Returns
        -------
        bool
            True, if the XModelNode objects are removed successfully; otherwise, False.
        """
        if xnodes is None or len(xnodes) == 0:
            return True

        for xnode in xnodes:
            if isinstance(xnode, XModelNode):
                if not self.remove_xnode(xnode):
                    return False
            else:
                return False
        return True

    def infer_shape(
        self, layout: Layout, disable_pbar: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Perform shape inference based on the current layout and inputs shape.

        Parameters
        ----------
        layout : Layout
            layout setting.
        disable_pbar : bool, optional
            disable the show of the progress bar, by default False

        Returns
        -------
        Tuple[bool, Optional[str]]
            True, if the shape inference is done successfully; otherwise, False.
        """
        assert isinstance(layout, Layout), "'layout' should be a Layout enum value."
        assert isinstance(
            disable_pbar, bool
        ), "'disable_pbar' should be a boolean value."

        if self.xnodes is None or len(self.xnodes) == 0:
            return True

        # sort xnodes
        self.topsort()

        # perform shape inference with the current layout if the model does not perform
        ok, error = self.__do_shape_inference(Layout[self.layout])
        if not ok:
            return ok, error

        if self.layout != layout.name:
            ok, error = self.__do_shape_inference(layout)
            if not ok:
                return ok, error
            # update layout
            self.layout = layout.name

        return True, None

    def __do_shape_inference(
        self, layout: Layout, disable_pbar: bool = False
    ) -> Tuple[bool, Optional[str]]:
        pbar = tqdm(
            self.xnodes,
            desc=f"[INFO] infer shape ({layout.name})",
            bar_format="{desc:27}:{percentage:3.0f}%|{bar}{r_bar:50}",
            disable=disable_pbar,
        )
        for xnode in pbar:
            # infer each xnode
            try:
                xnode.infer_shape(layout)
            except NotImplementedError as _:
                return (
                    False,
                    f"[ERROR] Not implement infer interface: xnnc node kind: {xnode.op_type}",
                )

            # update inputs_tensor of child nodes
            if xnode.top is not None and len(xnode.top) > 0:
                for i, cname in enumerate(xnode.top):
                    cnode = self.get_xnode_by_name(cname)
                    assert cnode is not None, f"[ERROR] Not found child node: {cname}"

                    # update inputs_tensor of child nodes
                    if cnode.inputs_tensor is None or len(cnode.inputs_tensor) == 0:
                        cnode.inputs_tensor = [None] * len(cnode.bottom)

                    try:
                        for idx in range(len(cnode.bottom)):
                            if cnode.bottom[idx] == xnode.op_name:
                                cnode.inputs_tensor[idx] = xnode.outputs_tensor[
                                    0
                                ].copy()

                                # set sequences of dims of inputs
                                curr_seq_dict = cnode.sequence_dims.get(layout.name)
                                if (
                                    curr_seq_dict["in"] is None
                                    or len(curr_seq_dict["in"]) == 0
                                ):
                                    curr_seq_dict["in"] = [None] * len(cnode.bottom)

                                assert (
                                    xnode.sequence_dims.get(layout.name).get("out")
                                    is not None
                                ), f"[ERROR] Failed to get dimension trace info of outputs: type: {xnode.op_type}, name: {xnode.op_name}."

                                curr_seq_dict["in"][idx] = xnode.sequence_dims.get(
                                    layout.name
                                ).get("out")[0]
                                assert (
                                    curr_seq_dict["in"][idx] is not None
                                ), f"[ERROR] Failed to set dimension sequence of inputs for child node: type: {cnode.op_type}, name: {cnode.op_name}. Current node: type: {xnode.op_type}, name: {xnode.op_name}."

                    except IndexError as e:
                        return (
                            False,
                            f"[IndexError] op type: {xnode.op_type}, name: {xnode.op_name}; child op type: {cnode.op_type}, name: {cnode.op_name}",
                        )

        self.__shape_inferred = True
        return True, None

    def change_layout(self, new_layout: str) -> bool:
        """
        Change the layout of the XModel object, including its nodes. The layout change operation also triggers the shape inference based on the new layout.

        Parameters
        ----------
        new_layout : str
            new layout value.

        Returns
        -------
        bool
            True, if the change is made successfully; otherwise, False.
        """
        assert new_layout is not None, "'new_layout' should not be None."
        assert isinstance(new_layout, str), "'new_layout' should be of str type."
        new_layout == new_layout.upper()
        assert new_layout in [
            "NCHW",
            "NHWC",
        ], f"'new_layout' should be one of 'NCHW' and 'NHWC': actual: {new_layout}"

        if new_layout == self.layout:
            return True
        else:
            self.layout = new_layout

        if self.xnodes is None or len(self.xnodes) == 0:
            return True

        input_xnodes = []
        for xnode in self.xnodes:
            if isinstance(xnode, XModelNodeInput):
                input_xnodes.append(xnode)
        assert (
            len(input_xnodes) > 0
        ), "[ERROR] Not found input nodes in current XModel instance."
        for xnode in input_xnodes:
            in_shape = xnode.shape
            if self.layout == "NCHW":
                N, H, W, C = in_shape
                in_shape = [N, C, H, W]
            else:
                N, C, H, W = in_shape
                in_shape = [N, H, W, C]
            xnode.shape = in_shape
            if xnode.init_layout != xnode.layout:
                xnode.init_layout = xnode.layout
            xnode.layout = new_layout

        # infer shape of each subsequent nodes
        try:
            self.infer_shape()
        except Exception as _:
            print(
                f"[ERROR] Failed to perform shape inference while changing the layout."
            )
            return False
        return True

    def has_xnode(self, name: str) -> bool:
        """
        Check if an XModelNode object with the specified name exists or not.

        Parameters
        ----------
        name : str
            name of the target XModelNode.

        Returns
        -------
        bool
            True, if an XModelNode object with the specified name exists; otherwise, False.
        """
        return name in self.__xnode_dict

    def rename_xnode(self, xnode: XModelNode, new_name: str) -> bool:
        """
        Update name of an XModelNode object.

        Parameters
        ----------
        xnode : XModelNode
            the target XModelNode object.
        new_name : str
            new name.

        Returns
        -------
        bool
            True, if the name is updated successfully; otherwise, False.
        """
        assert xnode is not None, "'xnode' should not be None."
        assert isinstance(
            xnode, XModelNode
        ), "'xnode' should be of XModelNode or its subclass type."
        assert new_name is not None, "'new_name' should not be None."
        assert isinstance(new_name, str), "'new_name' should be of str type."

        if not self.has_xnode(xnode.op_name):
            return False

        # update the output of parent nodes
        if xnode.bottom is not None and len(xnode.bottom) > 0:
            for pname in xnode.bottom:
                pnode = self.get_xnode_by_name(pname)
                assert pnode is not None, f"[ERROR] Not found parent node: {pname}"
                idx = pnode.top.index(xnode.op_name)
                pnode.top[idx] = new_name

        # update the input of child nodes
        if xnode.top is not None and len(xnode.top) > 0:
            for cname in xnode.top:
                cnode = self.get_xnode_by_name(cname)
                assert cnode is not None, f"[ERROR] Not found child node: {cname}"
                idx = cnode.bottom.index(xnode.op_name)
                cnode.bottom[idx] = new_name

        # update op name with new_name
        del self.__xnode_dict[xnode.op_name]
        xnode.op_name = new_name
        self.__xnode_dict[new_name] = xnode

        return True

    def predecessors(self, xnode: XModelNode) -> List[XModelNode]:
        """
        Get predecessors of an XModelNode object.

        Parameters
        ----------
        xnode : XModelNode
            the target XModelNode object.

        Returns
        -------
        List[XModelNode]
            list of XModelNode objects.
        """
        pred = []
        if xnode.bottom is not None and len(xnode.bottom) > 0:
            for pname in xnode.bottom:
                pnode = self.get_xnode_by_name(pname)
                assert pnode is not None, f"[ERROR] Not found predecessor node: {pname}"
                pred.append(pnode)
        return pred

    def successors(self, xnode: XModelNode) -> List[XModelNode]:
        """
        Get successors of an XModelNode object.

        Parameters
        ----------
        xnode : XModelNode
            the target XModelNode object.

        Returns
        -------
        List[XModelNode]
            list of XModelNode objects.
        """
        succ = []
        if xnode.top is not None and len(xnode.top) > 0:
            for cname in xnode.top:
                cnode = self.get_xnode_by_name(cname)
                assert (
                    cnode is not None
                ), f"[ERROR] Not found successor node: {cname}. node name: {xnode.op_name}, type: {xnode.op_type}"
                succ.append(cnode)
        return succ

    def number_of_edges(self, source: XModelNode, target: XModelNode) -> int:
        """
        Get the number of the direct edges between two nodes.

        Parameters
        ----------
        xnode1 : XModelNode
            source XModelNode object
        xnode2 : XModelNode
            target XModelNode object

        Returns
        -------
        int
            number of the direct edges between the specified two nodes.
        """
        assert source is not None, "'xnode1' should not be None."
        assert isinstance(source, XModelNode), "'xnode1' should be of XModelNode type."
        assert target is not None, "'xnode2' should not be None."
        assert isinstance(target, XModelNode), "'xnode2' should be of XModelNode type."

        # xnode1 == xnode2
        if source == target:
            return 0
        # xnode2 -> xnode1
        pred1 = self.predecessors(source)
        if pred1 and target in pred1:
            succ2 = self.successors(target)
            assert source in succ2
            return 1
        # xnode1 -> xnode2
        succ1 = self.predecessors(source)
        if succ1 and target in succ1:
            pred2 = self.predecessors(target)
            assert source in pred2
            return 1

        return 0

    def render(
        self, label="kind", filename=None, directory=None, view=False, cleanup=False
    ) -> NoReturn:
        """
        Visualize an XModel instance with the Graphviz engine.

        Parameters
        ----------
        label : str, optional
            node label, by default "kind"
        filename : [type], optional
            file name for saving the source, by default None
        directory : [type], optional
            (sub)directory for source saving and rendering, by default None
        view : bool, optional
            open the rendered result with the default application, by default False
        cleanup : bool, optional
            delete the source file after rendering, by default False

        Returns
        -------
        NoReturn
            No return value.
        """
        assert len(self.__xnode_dict) > 0

        label = label.lower()
        assert (
            label in ["name", "kind", "layout_type"] or label is None
        ), "'label' should be 'kind', 'name', 'layout_type' or None."

        # create a DG instance with svg format
        graph = graphviz.Digraph(format="svg")

        # create nodes for graph
        for _, node in self.__xnode_dict.items():
            label_txt = node.op_type
            if label == "name":
                label_txt = node.op_name + "\nkind:" + label_txt
            elif label == "layout_type":
                if node.layout_type == LayoutType.INSENSITIVE:
                    kind = "I"
                elif node.layout_type == LayoutType.TOLERANT:
                    kind = "T"
                elif node.layout_type == LayoutType.DEPENDENT:
                    kind = "D"
                else:
                    kind = "R"
                if node.op_type in ["concat", "softmax", "scale"]:
                    label_txt = f"{label_txt}\n{kind}\n{node.sequence_dims[node.layout]['in']}\naxis:{node.axis}\n{node.sequence_dims[node.layout]['out']}"
                elif node.op_type in ["permute"]:
                    label_txt = f"{label_txt}\n{kind}\n{node.sequence_dims[node.layout]['in']}\norder:{node.order}\n{node.sequence_dims[node.layout]['out']}"
                elif node.op_type in ["flatten"]:
                    label_txt = f"{label_txt}\n{kind}\n{node.sequence_dims[node.layout]['in']}\nstart_dim:{node.start_dim}\nend_dim:{node.end_dim}\n{node.sequence_dims[node.layout]['out']}"
                else:
                    label_txt = f"{label_txt}\n{kind}\n{node.op_name}\n{node.sequence_dims[node.layout]['in']}\n{node.sequence_dims[node.layout]['out']}"
                # label_txt = label_txt + "\n" + kind
            graph.node(
                node.op_name,
                label=label_txt,
            )

        # create edges for graph
        for _, node in self.__xnode_dict.items():
            if len(node.top) > 0:
                for cname in node.top:
                    edge_label = None
                    if (
                        node.outputs_tensor is not None
                        and len(node.outputs_tensor) == 1
                    ):
                        edge_label = f"{node.outputs_tensor[0].shape}\ndtype: {node.outputs_tensor[0].dtype.name}\nlayout: {node.layout}\nseq_dims: {node.sequence_dims[node.layout]['out']}"
                    graph.edge(node.op_name, cname, label=edge_label)

        # render and save the graph
        if filename is None:
            filename = self.name
        graph.render(filename, directory, view, cleanup)

    def clear(self) -> NoReturn:
        """
        Clear up all nodes.

        Returns
        -------
        NoReturn
            No return value.
        """
        self.__xnode_dict = {}

    def serialize(self, fname: Path, target: TargetType) -> bool:
        """
        Serialize xnnc XModel object to proto file of specified type.

        Parameters
        ----------
        fname : Path
            proto file path to save.
        target : TargetType
            the type of proto file.

        Returns
        -------
        bool
            True, if the target proto file is generated successfully; otherwise, False.
        """
        assert fname is not None and isinstance(
            fname, Path
        ), "'fname' should be a Path object."
        assert isinstance(
            target, TargetType
        ), "'target' should be a TargetType enum value."

        if not self.xnodes:
            return False

        print(f"[INFO] frozen model ...", end="\r")

        # create an open graph
        open_graph = openir.GraphProto()

        # graph properties
        open_graph.name = self.name
        open_graph.origin = self.origin
        open_graph.layout = openir.Layout.Value(self.layout.upper())

        nodes = []
        pbar = tqdm(
            self.xnodes,
            desc="[INFO] serialize model",
            bar_format="{desc:27}:{percentage:3.0f}%|{bar}{r_bar:50}",
        )
        for xnode in pbar:
            nodes.append(XModelNode.serialize(xnode, target))

        open_graph.nodes.extend(nodes)

        # serialize
        with open(fname, "wb") as f:
            f.write(open_graph.SerializeToString())

        if fname.exists():
            print(f"[INFO] frozen model: {fname.resolve()}")
            return True
        return False

    @staticmethod
    def deserialize(fname: Path) -> "XModel":
        """
        Deserialize openir proto file to xmodel object.

        Returns
        -------
        xnnc.ir.XModel
            xnnc XModel object.

        Raises
        ------
        NotImplementedError
            If any XModelNode object in deserialized XModel object does not implement deserialize interface, raise a NotImplementedError exception.
        """
        assert fname is not None, "'fname' should not be None."
        assert fname.exists(), f"[ERROR] Not found: {str(fname)}"

        # load model file
        graph = openir.GraphProto()
        with open(fname, "rb") as pf:
            graph.ParseFromString(pf.read())

        xmodel = XModel(graph.name, graph.origin)
        xmodel.layout = openir.Layout.Name(graph.layout)

        # iterate each node
        pbar = tqdm(
            graph.nodes,
            desc="[INFO] deserialize model",
            bar_format="{desc:27}:{percentage:3.0f}%|{bar}{r_bar:50}",
        )
        for node in pbar:
            xnode_class = REGISTERED_OPS.get(node.WhichOneof("params"))
            if xnode_class is None:
                xnode_class = REGISTERED_OPS.get(node.kind)
                assert (
                    xnode_class is not None
                ), f"[ERROR] Unsupported node type: {node.kind}."
            try:
                xnode = xnode_class.deserialize(node)
            except NotImplementedError:
                raise NotImplementedError(
                    f"[ERROR] Not implement deserialize interface in {str(xnode_class)}."
                )
            xmodel.add_xnode(xnode)

        return xmodel
