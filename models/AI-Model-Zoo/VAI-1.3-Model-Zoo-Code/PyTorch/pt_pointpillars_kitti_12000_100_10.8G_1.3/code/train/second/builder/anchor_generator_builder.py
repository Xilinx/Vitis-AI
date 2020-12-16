
# This code is based on: https://github.com/nutonomy/second.pytorch.git
# 
# MIT License
# Copyright (c) 2018 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

from second.protos import box_coder_pb2
from second.core.anchor_generator import (
    AnchorGeneratorStride, AnchorGeneratorRange)


def build(anchor_config):
    """Create optimizer based on config.

  Args:
    optimizer_config: A Optimizer proto message.

  Returns:
    An optimizer and a list of variables for summary.

  Raises:
    ValueError: when using an unsupported input data type.
  """
    ag_type = anchor_config.WhichOneof('anchor_generator')

    if ag_type == 'anchor_generator_stride':
        config = anchor_config.anchor_generator_stride
        ag = AnchorGeneratorStride(
            sizes=list(config.sizes),
            anchor_strides=list(config.strides),
            anchor_offsets=list(config.offsets),
            rotations=list(config.rotations),
            match_threshold=config.matched_threshold,
            unmatch_threshold=config.unmatched_threshold,
            class_id=config.class_name)
        return ag
    elif ag_type == 'anchor_generator_range':
        config = anchor_config.anchor_generator_range
        ag = AnchorGeneratorRange(
            sizes=list(config.sizes),
            anchor_ranges=list(config.anchor_ranges),
            rotations=list(config.rotations),
            match_threshold=config.matched_threshold,
            unmatch_threshold=config.unmatched_threshold,
            class_id=config.class_name)
        return ag
    else:
        raise ValueError(" unknown anchor generator type")