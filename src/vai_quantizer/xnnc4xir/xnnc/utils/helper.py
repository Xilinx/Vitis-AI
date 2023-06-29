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

from pathlib import Path
from enum import Enum, auto
import graphviz


def copy_data(src_layer, dst_layer):
    """
    Copy items from src_layer into dst_layer, which are in src_layer, but not in dst_layer
    :param src_layer: a source dict
    :param dst_layer: a destination dict
    """
    assert src_layer is not None
    assert dst_layer is not None
    assert (
        src_layer["name"].lower() == dst_layer["name"].lower()
    ), "src name: {0}, src type: {1}, dst name: {2}, dst type: {3}".format(
        src_layer["name"].lower(),
        src_layer["type"].lower(),
        dst_layer["name"].lower(),
        dst_layer["type"].lower(),
    )

    for key, value in src_layer.items():
        if key not in dst_layer and key in ["blobs", "fixedParam"]:
            dst_layer[key] = value


def check_filepath(file_path: Path, extension: str = None) -> (bool, str, Path):
    """
    Check if the specified file path is valid. If extension is specified, also check if the file contained
    in the file path has the same extension name.

    Parameters:
        file_path: Path, an instance of Path indicating the path of a file.
        extension: str, the extension name the file should have.

    Return: (flag, error_msg, file_path):
        flag:      bool, indicates if the file path is valid or not.
        error_msg: str or None, indicates the error message if flag is False; otherwise, None.
        file_path: Path, the file path
    """
    # if not file_path.is_dir():
    #     return False, "Invalid dir: {0}".format(file_path), file_path
    if not file_path.is_absolute():
        file_path = file_path.absolute()
    if not file_path.is_file():
        return (
            False,
            "The file path does not contain a valid file: {0}".format(file_path),
            file_path,
        )
    else:
        if extension is not None and file_path.suffix != extension:
            return (
                False,
                "The extension of the file should be '{0}': {1}".format(
                    extension, file_path
                ),
                file_path,
            )
    return True, "", file_path


class Layout(Enum):
    NCHW = auto()
    NHWC = auto()


def render_xmodel(xmodel, filename=None, directory=None, view=False, cleanup=False):
    """
    Visualize an XModel instance with the Graphviz engine.

    Parameters:
        - xgraph: an XModel instance
        - filename: Filename for saving the source.
        - directory: (Sub)directory for source saving and rendering.
        - view: Open the rendered result with the default application.
        - cleanup: Delete the source file after rendering.
    """
    assert xmodel is not None
    assert len(xmodel.xnodes) > 0

    # create a DG instance with svg format
    graph = graphviz.Digraph(format="svg")

    # create nodes for graph
    for node in xmodel.xnodes:
        graph.node(node.op_name, label=node.op_type)

    # create edges for graph
    for node in xmodel.xnodes:
        if len(node.top) > 0:
            for cname in node.top:
                label = None
                if node.__class__.__name__ == "XModelNodeInput":
                    label = str(node.shape)
                elif len(node.outputs_tensor_shape) > 0:
                    label = str(node.outputs_tensor_shape[0])
                graph.edge(node.op_name, cname, label=label)

    # render and save the graph
    if filename is None:
        filename = xmodel.name
    graph.render(filename, directory, view, cleanup)


def render_xmodel_opt(xmodel, filename=None, directory=None, view=False, cleanup=False):
    """
    Visualize an optimized XModel instance with the Graphviz engine.

    Parameters:
        - xgraph: an XModel instance
        - filename: Filename for saving the source.
        - directory: (Sub)directory for source saving and rendering.
        - view: Open the rendered result with the default application.
        - cleanup: Delete the source file after rendering.
    """
    assert xmodel is not None
    assert len(xmodel.xnodes) > 0

    # create a DG instance with svg format
    graph = graphviz.Digraph(format="svg")

    node_dict = {}
    candidates = []
    for node in xmodel.xnodes:
        node_dict[node.op_name] = node
        if len(node.bottom) == 0:
            candidates.append(node)

    fused_node_dict = {}

    while len(candidates) > 0:
        xnode = candidates.pop(0)
        graph_node_label = xnode.op_type
        # case 1: node having in-place child node
        if (
            not xnode.is_inplace
            and len(xnode.top) > 0
            and node_dict.get(xnode.top[0]).is_inplace
        ):
            # update graph node label
            for ip_name in xnode.top:
                op_node = node_dict.get(ip_name)
                graph_node_label = graph_node_label + "_" + op_node.op_type

            # get last non-in-place node in the fusion chain
            last_ip_name = xnode.top[-1]
            last_ip_node = node_dict.get(last_ip_name)
            assert last_ip_node is not None

            # put the names of starting and ending nodes in dict
            fused_node_dict[xnode.op_name] = last_ip_node.top

            # update candidates with non-in-place nodes
            for xname in last_ip_node.top:
                nip_node = node_dict.get(xname)
                assert nip_node is not None
                candidates.append(nip_node)

        # case 2: node not having non-in-place node
        else:
            fused_node_dict[xnode.op_name] = xnode.top

            # update candidates with non-in-place nodes
            for xname in xnode.top:
                nip_node = node_dict.get(xname)
                assert nip_node is not None
                candidates.append(nip_node)

        # create graph node
        graph.node(xnode.op_name, label=graph_node_label)

    # create edges for graph
    for sname, enames in fused_node_dict.items():
        for ename in enames:
            graph.edge(sname, ename)

    # render and save the graph
    if filename is None:
        filename = xmodel.name + "_opt"
    graph.render(filename, directory, view, cleanup)

