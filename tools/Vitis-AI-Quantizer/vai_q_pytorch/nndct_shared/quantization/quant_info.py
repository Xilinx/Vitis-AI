

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
from .commander import QuantConfigerCommander
from .quant_strategy import create_quant_strategy


class QuantInfoMgr(NndctGraphHolder):

  def __init__(self, graph, model_type, bitw, bita, lstm, mix_bit): 
    super().__init__(graph, model_type)
    self._QuantGroups = None

    self.group_graph()
    quant_strategy = create_quant_strategy(bitw, bita, lstm, mix_bit)
    self._quant_info = quant_strategy.create_quant_config(self)

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
  
