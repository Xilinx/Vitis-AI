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

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, NoReturn

from xnnc.ir.xmodel import XModel
from xnnc.ir.xnode import XModelNode
from xnnc.ir.enums import Layout

# create logger
logger = logging.getLogger(__name__)

curr_path = Path(__file__).absolute().parent
PATTERN_FILE = curr_path / "patterns.json"


class OptManager(object):
    pattern_graphs: Dict[str, List[XModel]] = {}

    @classmethod
    def dispatch(cls, xmodel: XModel, level: str) -> NoReturn:
        assert xmodel is not None, "'xmodel' should not be None."
        assert level is not None and level in [
            "xnnc",
            "xir",
        ], f"'level' should be a string, which is one of 'xnnc' and 'xir' (case-sensitive)."

        origin = xmodel.origin

        if not xmodel.shape_inferred:
            xmodel.infer_shape(layout=Layout[xmodel.layout])

        # load patterns
        pattern_dict = cls.__load_patterns(origin, level)

        # * call specific optimizer

        mod_name: str = origin + "_optimizer"
        mod = importlib.import_module("xnnc.optimizer." + mod_name)
        if mod is None:
            logger.info("Not found the target module: {0}".format(mod_name))
            sys.exit(1)
        # specify optimizer class name
        class_name: str = None
        if origin == "caffe":
            class_name = "CaffeOptimizer"
        elif origin == "tensorflow":
            class_name = "TFOptimizer"
        elif origin == "tensorflow2":
            class_name = "TF2Optimizer"
        elif origin == "pytorch":
            class_name = "PyTorchOptimizer"
        elif origin == "nndct":
            class_name = "NNDCTOptimizer"
        else:
            raise ValueError(f"[ERROR] Unsupported original model type: {origin}")
        logger.debug(f"model type: {origin}, optimizer type: {class_name}")

        optimizer = None
        if hasattr(mod, class_name):
            optimizer = getattr(mod, class_name)
        if optimizer is None:
            logger.info(f"{mod} has no class named {class_name}.")
            sys.exit(1)
        optimizer.run(xmodel, pattern_dict)

    @classmethod
    def __load_patterns(cls, origin: str, level: str) -> Dict[str, Any]:
        """Load graph patterns from patterns.json

        Parameters
        ----------
        origin : Optional[str], optional
            type of original model, by default None
        """
        assert origin is not None and origin in [
            "caffe",
            "tensorflow",
            "tensorflow2",
            "pytorch",
            "nndct",
        ], f"'origin' should be a string, which is one of 'caffe', 'tensorflow', 'tensorflow2', 'pytorch', 'nndct' (case-sensitive)."
        assert level is not None and level in [
            "xnnc",
            "xir",
        ], f"'level' should be a string, which is one of 'xnnc' and 'xir' (case-sensitive)."

        def parse(patterns, kind):
            graphs = []
            for ptn in patterns:
                if ptn["visible"]:
                    name = ptn.get("name")
                    assert name is not None
                    # create pattern graph
                    graph = XModel(name, kind)
                    # add nodes
                    assert "nodes" in ptn
                    nodes = ptn["nodes"]
                    assert len(nodes) > 0
                    for node in nodes:
                        op_name = node.get("name")
                        assert op_name is not None
                        op_type = node.get("kind")
                        assert op_type is not None
                        xnode = XModelNode(op_name, op_type)
                        graph.add_xnode(xnode)
                    # add edges
                    assert "edges" in ptn
                    edges = ptn["edges"]
                    if len(edges) > 0:
                        for edge in edges:
                            src = edge.get("src")
                            assert src is not None
                            assert graph.has_xnode(src)
                            src_node = graph.get_xnode_by_name(src)
                            dst = edge.get("dest")
                            assert dst is not None
                            assert graph.has_xnode(dst)
                            dst_node = graph.get_xnode_by_name(dst)
                            src_node.top.append(dst)
                            dst_node.bottom.append(src)

                    graphs.append(graph)
            return graphs

        with open(PATTERN_FILE.absolute(), "r") as f:
            pattern_config = json.load(f)

        # construct pattern graphs
        graphs = None
        if origin in pattern_config and level in pattern_config[origin]:
            patterns = pattern_config[origin][level]
            if len(patterns) > 0:
                graphs = parse(patterns, origin)

        return {"origin": origin, "level": level, "patterns": graphs}
