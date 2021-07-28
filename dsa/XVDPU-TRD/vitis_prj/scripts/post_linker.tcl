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
set cell_noc {NOC_0}
set cell_dpu {DPUCVDX8G}
set cell_clk {clk_wiz_accel}
set pl_freq 333
set CPB      32

set cfg_dpu_batch           [get_property CONFIG.BATCH_N              [get_bd_cells $cell_dpu]]
set cfg_dpu_load_instr      {1}
set cfg_dpu_load_wgt        [get_property CONFIG.LOAD_PARALLEL_WGT    [get_bd_cells $cell_dpu]]
set cfg_dpu_load_img        [get_property CONFIG.LOAD_PARALLEL_IMG    [get_bd_cells $cell_dpu]]
set cfg_dpu_load_img_batch  [expr $cfg_dpu_load_img * $cfg_dpu_batch]
set cfg_dpu_load_bias       [get_property CONFIG.LOAD_PARALLEL_BIAS   [get_bd_cells $cell_dpu]]
set cfg_noc_num_si_pfm      {15}
set cfg_noc_num_si_dpu      [expr $cfg_dpu_load_instr + $cfg_dpu_load_wgt + $cfg_dpu_load_img_batch + $cfg_dpu_load_bias]
set cfg_noc_num_si          [expr $cfg_noc_num_si_pfm + $cfg_noc_num_si_dpu]
set cfg_noc_num_clk_pfm     {12}
set cfg_noc_num_clk_dpu     {1}
set cfg_noc_num_clk         [expr $cfg_noc_num_clk_pfm + $cfg_noc_num_clk_dpu]
set cfg_CPB16_num_ifm_ports [expr 8 * $cfg_dpu_batch]


#Set AIE core Freq
set_property -dict [list CONFIG.AIE_CORE_REF_CTRL_FREQMHZ {1333}] [get_bd_cells $cell_aie]

#NOC
set_property -dict [list CONFIG.MC_INPUT_FREQUENCY0 {201.501} CONFIG.MC_INPUTCLK0_PERIOD {4963} CONFIG.MC_F1_LPDDR4_MR1 {0x000} CONFIG.MC_F1_LPDDR4_MR2 {0x000}] [get_bd_cells $cell_noc]
set_property -dict [list CONFIG.NUM_SI $cfg_noc_num_si CONFIG.NUM_CLKS $cfg_noc_num_clk] [get_bd_cells $cell_noc]
set_property -dict [list CONFIG.MC_CHANNEL_INTERLEAVING {true} CONFIG.MC_CH_INTERLEAVING_SIZE {128_Bytes} CONFIG.MC_LPDDR4_REFRESH_TYPE {PER_BANK} CONFIG.MC_TRC {60000} CONFIG.MC_PRE_DEF_ADDR_MAP_SEL {ROW_BANK_COLUMN} CONFIG.MC_ADDR_BIT9 {CA6}] [get_bd_cells $cell_noc]

delete_bd_objs [get_bd_intf_nets axi_ic_$cell_noc\_S*_AXI_M*_AXI] [get_bd_cells axi_ic_$cell_noc\_S*_AXI] [get_bd_intf_nets $cell_dpu\_M*_axi]

#Create AXI's connection
proc bip_connect_dpu_axi_to_noc { cell_dpu cell_noc list_base mi_name mi_total mc_read_bw } {
  set si_base           [lindex $list_base 0]
  set mc_base           [lindex $list_base 1]
  set si_list           [lindex $list_base 2]
  set cfg_noc_num_mcp   [get_property CONFIG.NUM_MCP  [get_bd_cells $cell_noc]]
  for {set i 0} {$i < $mi_total} {incr i} {
    set si_i      [format %02d [expr $i+$si_base]]
    set mi_i      [format %02d       $i          ]
    set mc_i      MC_[expr ($mc_base + $i) % $cfg_noc_num_mcp]
    set noc_axi   [get_bd_intf_pins $cell_noc/S$si_i\_AXI]
    set dpu_axi   [get_bd_intf_pins $cell_dpu/M$mi_i\_$mi_name\_AXI]
    set prop      "CONFIG.CONNECTIONS { $mc_i { read_bw {$mc_read_bw} write_bw {32} read_avg_burst {4} write_avg_burst {4} }}"
    if { [string toupper $mi_name] == {WGT} } {
      lappend prop CONFIG.R_TRAFFIC_CLASS {LOW_LATENCY}
    }
    set_property -dict $prop $noc_axi
    connect_bd_intf_net $dpu_axi $noc_axi
    lappend si_list S$si_i\_AXI
  }
  return [list [expr  $si_base + $mi_total] [expr ($mc_base + $mi_total) % $cfg_noc_num_mcp] $si_list ]
}

set list_base [list $cfg_noc_num_si_pfm 0 {} ]
set list_base [bip_connect_dpu_axi_to_noc $cell_dpu $cell_noc $list_base instr $cfg_dpu_load_instr      64    ]
set list_base [bip_connect_dpu_axi_to_noc $cell_dpu $cell_noc $list_base wgt   $cfg_dpu_load_wgt        6000  ]
set list_base [bip_connect_dpu_axi_to_noc $cell_dpu $cell_noc $list_base img   $cfg_dpu_load_img_batch  1000  ]
set list_base [bip_connect_dpu_axi_to_noc $cell_dpu $cell_noc $list_base bias  $cfg_dpu_load_bias       1000  ]

set pin_noc_dpu aclk[expr $cfg_noc_num_clk - 1]
set_property -dict [list CONFIG.ASSOCIATED_BUSIF [join [lindex $list_base 2] : ]] [get_bd_pins $cell_noc/$pin_noc_dpu ]

#Set clock freq
set l_clk_wiz_freq [split [get_property CONFIG.CLKOUT_REQUESTED_OUT_FREQUENCY [get_bd_cells $cell_clk]] ","]
set l_clk_wiz_port [split [get_property CONFIG.CLKOUT_PORT                    [get_bd_cells $cell_clk]] ","]
lset l_clk_wiz_freq 0 $pl_freq
set pin_clk_out [lindex $l_clk_wiz_port 0]
set_property -dict [list CONFIG.CLKOUT_REQUESTED_OUT_FREQUENCY  [join $l_clk_wiz_freq , ]] [get_bd_cells $cell_clk]

#Delete default clock and reset connection
disconnect_bd_net /net_mb_ss_0_clk_out2 [get_bd_pins $cell_aie/aclk0]
disconnect_bd_net /net_mb_ss_0_clk_out2 [get_bd_pins $cell_dpu/m_axi_aclk]
disconnect_bd_net /net_mb_ss_0_clk_out2 [get_bd_pins $cell_noc/$pin_noc_dpu]
disconnect_bd_net /net_mb_ss_0_dcm_locked [get_bd_pins $cell_dpu/m_axi_aresetn]

#Creat new clock and reset connection
connect_bd_net [get_bd_pins clk_wiz_accel/$pin_clk_out] [get_bd_pins $cell_aie/aclk0]
connect_bd_net [get_bd_pins clk_wiz_accel/$pin_clk_out] [get_bd_pins $cell_dpu/m_axi_aclk] 
connect_bd_net [get_bd_pins clk_wiz_accel/$pin_clk_out] [get_bd_pins $cell_noc/$pin_noc_dpu]
connect_bd_net [get_bd_pins rst_processor_pl_333Mhz/peripheral_aresetn] [get_bd_pins $cell_dpu/m_axi_aresetn] 
