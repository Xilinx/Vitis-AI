#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
########################################
# constrains for S_AXI_CLK_INDEPENDENT #
########################################

set clk_rs  [get_clocks -of_objects [get_ports s_axi_aclk]]
set clk_1x  [get_clocks -of_objects [get_ports m_axi_aclk ] ]
expr {($clk_rs!=$clk_1x)?[set_false_path -from $clk_1x  -to $clk_rs]:{}}
expr {($clk_rs!=$clk_1x)?[set_false_path -from $clk_rs  -to $clk_1x]:{}}