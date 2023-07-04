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
import logging
import sys
from pathlib import Path
from typing import List, Optional

from xnnc.ir.xmodel import XModel
from xnnc.ir.enums import Layout

# from xnnc.utils import helper
# from xnnc.utils.helper import Layout

# create logger
logger = logging.getLogger(__name__)


class CORE(object):
    @staticmethod
    def make_xmodel(
        model_files: List[Path],
        model_type: str,
        layout: Layout = Layout.NHWC,
        in_shapes: Optional[List[List[int]]] = None,
        batchsize: int = 1,
    ) -> XModel:
        """Generate an XModel instance from specified model files.

        Parameters
        ----------
        object : List[Path]
            list of paths to neural network models, which are prototxt and
            caffemodel files for Caffe, frozen pb file for TensorFlow, pth file for PyTorch,
            and pb file for ONNX.
        model_type : str
            the original model type: Caffe, TensorFlow, or ONNX.
        layout : Layout, optional
            the data format used by original model: "NCHW" or "NHWC"., by default Layout.NHWC
        in_shapes: Optional[List[List[int]]]
            shape info of the input feature maps.
        batchsize : int
            target batch size, by default 1.

        Returns
        -------
        XModel
            XModel object.
        """
        assert model_files is not None, "'model_files' should not be None."

        model_t: str = model_type.lower()
        if model_t not in [
            "caffe",
            "tensorflow",
            "tensorflow2",
            "pytorch",
            "nndct",
            "onnx",
        ]:
            error = f"[ERROR] 'model_type' shoud be one of 'caffe', 'tensorflow', 'tensorflow2', 'pytorch', 'nndct': actual: {model_t}."
            print(error)
            logger.info(error)
            sys.exit(1)

        if model_t == "tensorflow2":
            mod_name = "tensorflow_translator"
        elif model_t == "nndct":
            mod_name = "nndct_translator"
        else:
            mod_name: str = model_t + "_translator"
        mod = importlib.import_module("xnnc.translator." + mod_name)
        if mod is None:
            error = f"[ERROR] Not found the target module: {mod_name}"
            print(error)
            logger.info(error)
            sys.exit(1)

        # specify translator class name
        class_name: str = None
        if model_t == "caffe":
            class_name = "CaffeTranslator"
        elif model_t in ["tensorflow", "tensorflow2"]:
            class_name = "TFTranslator"
        elif model_t == "pytorch":
            class_name = "PyTorchTranslator"
        elif model_t == "nndct":
            class_name = "NNDCTTranslator"
        elif model_t == "onnx":
            class_name = "ONNXTranslator"
        else:
            raise ValueError(
                f"[ERROR] Not found xnnc translator for {class_name} model."
            )
        logger.debug(f"model type: {model_t}, translator type: {class_name}")

        translator = None
        if hasattr(mod, class_name):
            translator = getattr(mod, class_name)
        if translator is None:
            logger.info("{0} has no class named {1}.".format(mod, class_name))
            sys.exit(1)
        logger.info(f"start: translate raw model to xmodel")
        xmodel = translator.to_xmodel(
            model_files,
            layout,
            in_shapes=in_shapes,
            batchsize=batchsize,
            model_type=model_t,
        )
        logger.info(f"end: translate raw model to xmodel")

        return xmodel
