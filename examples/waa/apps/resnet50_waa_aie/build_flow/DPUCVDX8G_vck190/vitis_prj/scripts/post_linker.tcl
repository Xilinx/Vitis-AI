# /*******************************************************************************
# /*                                                                         
# * Copyright 2019 Xilinx Inc.                                               
# *                                                                          
# * Licensed under the Apache License, Version 2.0 (the "License");          
# * you may not use this file except in compliance with the License.         
# * You may obtain a copy of the License at                                  
# *                                                                          
# *    http://www.apache.org/licenses/LICENSE-2.0                            
# *                                                                          
# * Unless required by applicable law or agreed to in writing, software      
# * distributed under the License is distributed on an "AS IS" BASIS,        
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# * See the License for the specific language governing permissions and      
# * limitations under the License.                                           
# */
# *******************************************************************************/

set cell_aie {ai_engine_0}
set cell_noc {noc_lpddr4}	
set cfg_noc_num_mcp   [get_property CONFIG.NUM_MCP  [get_bd_cells $cell_noc]]
set noc_saxi_list [get_bd_intf_pins $cell_noc/S??_AXI]

set CPB      32
set cell_dpu {DPUCVDX8G_1}
set cfg_dpu_batch           [get_property CONFIG.BATCH_N              [get_bd_cells $cell_dpu]]
set cfg_CPB16_num_ifm_ports [expr 8 * $cfg_dpu_batch]

#For CPB=16, delete unexpected IP between AIE and XVDPU, and create direct connection for IFM AXIS ports.
if {$CPB == 16} {
   delete_bd_objs [get_bd_intf_nets DPUCVDX8G_m*_ifm_axis] [get_bd_intf_nets dwc_DPUCVDX8G_m*_ifm_axis_M_AXIS] [get_bd_cells cdc_DPUCVDX8G_m*_ifm_axis] [get_bd_cells dwc_DPUCVDX8G_m*_ifm_axis]
   #For IFM ports, create direct connection between DPU and AIE
   for {set i 0} {$i < $cfg_CPB16_num_ifm_ports} {incr i} {
       set j  [format %02d $i]
       connect_bd_intf_net [get_bd_intf_pins DPUCVDX8G/m$j\_ifm_axis] [get_bd_intf_pins ai_engine_0/S$j\_AXIS]
    }
}

#Set AIE core Freq
set_property -dict [list CONFIG.AIE_CORE_REF_CTRL_FREQMHZ {1250}] [get_bd_cells $cell_aie]

#Change setting of LPDDR
set_property -dict [list CONFIG.MC_CHANNEL_INTERLEAVING {true} CONFIG.MC_CH_INTERLEAVING_SIZE {128_Bytes} CONFIG.MC_LPDDR4_REFRESH_TYPE {PER_BANK} CONFIG.MC_TRC {60000} CONFIG.MC_ADDR_BIT9 {CA5}] [get_bd_cells $cell_noc]

#Set NOC Qos for XVDPU's AXI ports
foreach s_axi $noc_saxi_list { 
	set m_axi [get_bd_intf_pins -of [get_bd_intf_nets -of $s_axi] -filter { MODE == Master }]
	if [regexp WGT_AXI $m_axi] {
		set s_axi_num [expr [scan $s_axi "/$cell_noc/S%d_AXI"] + 1 ]
        set mc_i  MC_[expr $s_axi_num % $cfg_noc_num_mcp]
		set prop      "CONFIG.CONNECTIONS { $mc_i { read_bw {6000} write_bw {32} }} CONFIG.R_TRAFFIC_CLASS {LOW_LATENCY}"
		set_property -dict $prop $s_axi
	} elseif [regexp IMG_AXI|BIAS_AXI $m_axi] {
		set s_axi_num [expr [scan $s_axi "/$cell_noc/S%d_AXI"] + 1 ]
        set mc_i  MC_[expr $s_axi_num % $cfg_noc_num_mcp]
		set prop      "CONFIG.CONNECTIONS { $mc_i { read_bw {1000} write_bw {32} }}"
		set_property -dict $prop $s_axi
	} elseif [regexp INSTR_AXI $m_axi] {
		set s_axi_num [expr [scan $s_axi "/$cell_noc/S%d_AXI"] + 1 ]
        set mc_i  MC_[expr $s_axi_num % $cfg_noc_num_mcp]
		set prop      "CONFIG.CONNECTIONS { $mc_i { read_bw {64} write_bw {32} }}"
		set_property -dict $prop $s_axi
	}  
}  