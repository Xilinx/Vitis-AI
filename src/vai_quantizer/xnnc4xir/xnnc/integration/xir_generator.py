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
from typing import List, NoReturn, Optional

from xnnc.translator.tensorflow_translator import TFTranslator as tf_translator
from xnnc.utils.helper import Layout
from xnnc.xconverter import XConverter
from xnnc.optimizer import OptManager

# from xnnc.proto.tf_pb2.graph_pb2 import GraphDef


class XIRGenerator(object):
    @classmethod
    def from_tensorflow(
        cls,
        graph_def: "GraphDef",
        fname: Path,
        layout: Layout = Layout.NHWC,
        in_shapes: Optional[List[List[int]]] = None,
    ) -> NoReturn:
        """generate xmodel file from tensorflow frozen model"""
        assert graph_def is not None, f"'graph_def' should not be None."
        # assert isinstance(
        #     graph_def, GraphDef
        # ), f"'graph_def' should be of graph_pb2.GraphDef type."
        assert fname is not None, f"'fname' should not be None."
        assert isinstance(fname, Path), f"'fname' should be of Python Path type."
        assert fname.suffix == ".xmodel", f"'fname' should have a suffix of '.xmodel'."
        assert layout is not None, f"'layout' should not be None."
        assert isinstance(layout, Layout), f"'layout' should be of Layout type."

        if in_shapes is not None:
            assert isinstance(
                in_shapes, list
            ), "'in_shapes' should be of Python list type"
            assert len(in_shapes) > 0, "'in_shapes' should contain one or more entries."
            assert all(
                [isinstance(x, list) for x in in_shapes]
            ), "'in_shapes' should have entries of Python list type."

        # convert tensforflow frozen model into xnnc graph
        xmodel = tf_translator.create_xmodel(
            name="fake", layers=list(graph_def.node), layout=layout, in_shapes=in_shapes
        )
        assert (
            xmodel is not None
        ), f"[ERROR] Failed to convert the TensorFlow frozen model into xnnc model."

        # optimize xnnc graph
        OptManager.dispatch(xmodel, "xir")

        # convert xnnc graph into xir graph
        graph = XConverter.make_xir_graph(xmodel, layout)
        assert (
            graph is not None
        ), f"[ERROR] failed to convert xnnc model into xir graph."

        # serialize and dump xir graph as xmodel
        graph.serialize(fname)
        if fname.exists():
            print(f"[INFO] the generated xir model file at {fname.absolute()}")
            return True
        else:
            print(f"[ERROR] Failed to serialize the current xir model.")
            return False
