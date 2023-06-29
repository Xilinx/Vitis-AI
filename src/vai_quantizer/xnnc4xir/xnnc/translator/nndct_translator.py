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
from typing import List

from xnnc.ir.enums import Layout
from xnnc.ir.xmodel import XModel
from xnnc.ir.xnode import *
from xnnc.optimizer import OptManager
from xnnc.translator.base_translator import ITranslator

# create logger
logger = logging.getLogger(__name__)


class NNDCTTranslator(ITranslator):
    """
    Convert a NNDCT_PyTorch model into XModel object.
    """

    @classmethod
    def to_xmodel(
        cls, model_files: List[Path], layout: Layout = Layout.NCHW, *args, **kwargs
    ) -> XModel:
        assert (
            model_files is not None or len(model_files) == 0
        ), "'model_files' should contain a model file."
        assert isinstance(layout, Layout), f"'layout' should be of Layout enum type."

        # check model files
        if len(model_files) != 1:
            logger.error(
                "The 'model_files' argument should contain only one '.pb' file."
            )
            sys.exit(1)

        # load model architecture file
        nndct_model: Path = None
        if model_files[0].suffix == ".pb":
            nndct_model = model_files[0]
        assert (
            nndct_model is not None
        ), f"[ERROR] Specify a model with the '.pb' extension."
        assert nndct_model.exists(), f"[ERROR] Not found: {str(nndct_model)}"

        # deserialize
        xmodel = XModel.deserialize(nndct_model)

        # extract fixneuron
        cls.__extract_fixneuron(xmodel)

        xmodel.infer_shape(Layout.NHWC)

        # * perform platform-specific optimizations
        OptManager.dispatch(xmodel, "xnnc")

        return xmodel

    @classmethod
    def __extract_fixneuron(cls, xmodel: XModel) -> NoReturn:
        assert xmodel is not None, "'xmodel' should not be None."
        assert isinstance(xmodel, XModel), "'xmodel' should be of XModel type."

        if xmodel.size > 0:
            for xnode in xmodel.xnodes:
                if (
                    xnode.op_type != "fixneuron"
                    and xnode.quant_out["bit_width"] is not None
                    and xnode.quant_out["bit_width"] > 0
                ):
                    fixneuron = XModelNodeFixNeuron(xnode.op_name + "_fix")
                    fixneuron.quant_in["bit_width"] = fixneuron.quant_out[
                        "bit_width"
                    ] = xnode.quant_out["bit_width"]
                    fixneuron.quant_in["quantize_pos"] = fixneuron.quant_out[
                        "quantize_pos"
                    ] = xnode.quant_out["quantize_pos"]
                    fixneuron.quant_in["round_mode"] = fixneuron.quant_out[
                        "round_mode"
                    ] = xnode.quant_out["round_mode"]
                    fixneuron.quant_in["signed"] = fixneuron.quant_out[
                        "signed"
                    ] = xnode.quant_out["signed"]

                    # update bottom and top
                    fixneuron.bottom = [xnode.op_name]
                    fixneuron.top = [x for x in xnode.top]
                    xnode.top = [fixneuron.op_name]
                    # update bottom of child nodes
                    if len(fixneuron.top) > 0:
                        for cname in fixneuron.top:
                            cnode = xmodel.get_xnode_by_name(cname)
                            assert cnode is not None
                            idx = cnode.bottom.index(xnode.op_name)
                            cnode.bottom[idx] = fixneuron.op_name
                    xmodel.add_xnode(fixneuron)

                if (
                    xnode.op_type != "fixneuron"
                    and xnode.quant_in["bit_width"] is not None
                    and xnode.quant_in["bit_width"] > 0
                ):
                    fixneuron = XModelNodeFixNeuron(xnode.op_name + "_fix")
                    fixneuron.quant_in["bit_width"] = fixneuron.quant_out[
                        "bit_width"
                    ] = xnode.quant_in["bit_width"]
                    fixneuron.quant_in["quantize_pos"] = fixneuron.quant_out[
                        "quantize_pos"
                    ] = xnode.quant_in["quantize_pos"]
                    fixneuron.quant_in["round_mode"] = fixneuron.quant_out[
                        "round_mode"
                    ] = xnode.quant_in["round_mode"]
                    fixneuron.quant_in["signed"] = fixneuron.quant_out[
                        "signed"
                    ] = xnode.quant_in["signed"]

                    # update bottom and top
                    fixneuron.bottom = [x for x in xnode.bottom]
                    fixneuron.top = [xnode.op_name]
                    xnode.bottom = [fixneuron.op_name]
                    # update top of parent nodes
                    if len(fixneuron.bottom) > 0:
                        for pname in fixneuron.bottom:
                            pnode = xmodel.get_xnode_by_name(pname)
                            assert pnode is not None
                            idx = pnode.top.index(xnode.op_name)
                            pnode.top[idx] = fixneuron.op_name
                    xmodel.add_xnode(fixneuron)

            # topsort
            xmodel.topsort()
