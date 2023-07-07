

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

import copy
from nndct_shared.nndct_graph import NndctGraphHolder
from nndct_shared.utils import NndctScreenLogger
from .commander import QuantConfigerCommander
from .quant_strategy import create_quant_strategy
from nndct_shared.utils import NndctOption
from nndct_shared.base import NNDCT_OP
import pprint
pp = pprint.PrettyPrinter(indent=4)

class QuantInfoMgr(NndctGraphHolder):

  def __init__(self, graph, model_type, bitw, bita, lstm, mix_bit, custom_quant_ops=None): 
    super().__init__(graph, model_type)
    self._QuantGroups = None
    if custom_quant_ops:
      for op in custom_quant_ops:
        if op not in self.QUANTIZABLE_OPS:
          self.QUANTIZABLE_OPS.append(op)
          NndctScreenLogger().info(f"Convert `{op}` to quantizable op.")
          
    self.group_graph()
    quant_strategy = create_quant_strategy(bitw, bita, lstm, mix_bit)
    self._quant_info = quant_strategy.create_quant_config(self)
    if NndctOption.nndct_stat.value > 0:
      print('Quantization groups:')
      pp.pprint(self._QuantGroups)
      pp.pprint(self._quant_info)

    # check groups, only permit one quantizable node in one group in quant part
    ignored_list = [NNDCT_OP.SHAPE]
    for k, v in self._QuantGroups.items():
      if len(v) == 1:
        if len(self.Nndctgraph.parents(k)) == 0:
          break
      findQuantizableNode = False
      isIgnored = False
      type_list = self.LSTM_QUANTIZABLE_OPS if lstm else self.QUANTIZABLE_OPS
      for n in v:
        node = self.get_Nndctnode(node_name=n)
        if node.op.type in type_list:
          if findQuantizableNode:
            NndctScreenLogger().warning(f'Multiple quantizable node is found in group:')
            NndctScreenLogger().warning(f'{v}')
          else:
            findQuantizableNode = True
        elif node.op.type in ignored_list:
          isIgnored = True
      if not findQuantizableNode and not isIgnored:
        NndctScreenLogger().warning(f'No quantizable node is found in group, confirm no numerical calculation in the nodes:')
        NndctScreenLogger().warning(f'{v}')

  def group_graph(self):
    QuantConfigerCommander.register(self, 'scan_commander')
    commands = [k for k in self.scan_commander]
    quant_groups = {n.name: [n.name] for n in self.Nndctgraph.nodes if n.in_quant_part}
    while True:
      org_groups = copy.deepcopy(quant_groups)
      for c in commands:
        quant_groups = self.scan_commander[c](self.Nndctgraph, quant_groups)
      
      if org_groups == quant_groups:
        break
    for k, v in quant_groups.items():
      quant_groups[k] = sorted(v, key=lambda n: self.get_Nndctnode(n).idx)
    self._QuantGroups = quant_groups


  @property
  def quant_info(self):
    return self._quant_info
  
  @property
  def quant_groups(self):
    return self._QuantGroups
  
