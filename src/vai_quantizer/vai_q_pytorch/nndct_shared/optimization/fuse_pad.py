

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

from nndct_shared.base import NNDCT_OP


_OP_WITH_PAD = [
  NNDCT_OP.CONV2D,
  NNDCT_OP.DEPTHWISE_CONV2D,
  NNDCT_OP.CONVTRANSPOSE2D,
  NNDCT_OP.DEPTHWISE_CONVTRANSPOSE2D,
  NNDCT_OP.MAX_POOL,
  NNDCT_OP.AVG_POOL  
]

class PadFuseHandler(object):
  def __call__(self, *args, **kwargs):
    _, node_set = args
    pad_node = node_set[0]
    pad_mode = pad_node.node_attr(pad_node.op.AttrName.MODE)
    pad = pad_node.node_attr(pad_node.op.AttrName.PAD_WITH)
    pad_value = pad_node.node_attr(pad_node.op.AttrName.CONSTANT_VALUES)[0]
    if pad_mode != 0 or len(pad) != 8 or pad_value != 0 or any([pad[2 * i] != pad[2 * i + 1] for i in range(4)]) or any(cn.op.type not in _OP_WITH_PAD for cn in pad_node.owning_graph.children(pad_node)):
      pad_node.merged = False
      print(f"{pad_node.name} can't be fused")
      return

    for cn in pad_node.owning_graph.children(pad_node):
      cn_padding = cn.node_attr(cn.op.AttrName.PAD)
      cn_padding[0] += pad[4]
      cn_padding[1] += pad[5]
      cn_padding[2] += pad[2]
      cn_padding[3] += pad[3]
      cn.set_node_attr(cn.op.AttrName.PAD, cn_padding)

    pad_node.merged = True


    


