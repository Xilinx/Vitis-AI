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

# delete the fifo in the axi interconnect
set ps_cell [get_bd_cells * -hierarchical -quiet -filter {VLNV =~ "xilinx.com:ip:zynq_ultra_ps_e:*"}]
if {$ps_cell ne ""} {
	set pfm_ports_dict [get_property PFM.AXI_PORT $ps_cell]
	foreach {port} [get_bd_intf_pins $ps_cell/* -quiet -filter {MODE=~"Slave"}] {
		if {[dict exists $pfm_ports_dict [bd::utils::get_short_name $port]]} {
			set attached_interconnect [bd::utils::get_parent [find_bd_objs -relation connected_to $port ]]
			set intfCount [get_property CONFIG.NUM_SI $attached_interconnect]
			for {set i 0} {$i < $intfCount} {incr i} {
				set_property -dict [list CONFIG.S[format %02d $i]_HAS_REGSLICE {0} CONFIG.S[format %02d $i]_HAS_DATA_FIFO {0}] $attached_interconnect

			}
			set intfCount [get_property CONFIG.NUM_MI $attached_interconnect]
			for {set i 0} {$i < $intfCount} {incr i} {
				set_property -dict [list CONFIG.M[format %02d $i]_HAS_REGSLICE {0} CONFIG.M[format %02d $i]_HAS_DATA_FIFO {0}] $attached_interconnect
			}
		}
	}
}
set intr_cell [get_bd_cells * -hierarchical -quiet -filter {NAME =~ "interconnect_axifull"}]
if {$intr_cell ne ""} {
	set intfCount [get_property CONFIG.NUM_SI $intr_cell]
	for {set i 0} {$i < $intfCount} {incr i} {
		set_property -dict [list CONFIG.S[format %02d $i]_HAS_REGSLICE {0} CONFIG.S[format %02d $i]_HAS_DATA_FIFO {0}] $intr_cell
	}
	set intfCount [get_property CONFIG.NUM_MI $intr_cell]
	for {set i 0} {$i < $intfCount} {incr i} {
		set_property -dict [list CONFIG.M[format %02d $i]_HAS_REGSLICE {0} CONFIG.M[format %02d $i]_HAS_DATA_FIFO {0}] $intr_cell
	}
}


#set clk_out5,clk_out6 to 600Mhz, and enable the buffer with CE
set_property -dict [list \
		   CONFIG.CLKOUT6_REQUESTED_OUT_FREQ {600.000}\
		   CONFIG.CLKOUT6_DRIVES {Buffer_with_CE}\
		   CONFIG.CLKOUT7_DRIVES {Buffer_with_CE}\
		   CONFIG.MMCM_CLKOUT5_DIVIDE {2}\
		   CONFIG.CLKOUT1_JITTER {107.567}\
		   CONFIG.CLKOUT1_PHASE_ERROR {87.180}\
		   CONFIG.CLKOUT2_JITTER {94.862}\
		   CONFIG.CLKOUT2_PHASE_ERROR {87.180}\
		   CONFIG.CLKOUT3_JITTER {122.158}\
		   CONFIG.CLKOUT3_PHASE_ERROR {87.180}\
		   CONFIG.CLKOUT4_JITTER {115.831}\
		   CONFIG.CLKOUT4_PHASE_ERROR {87.180}\
		   CONFIG.CLKOUT5_JITTER {102.086}\
		   CONFIG.CLKOUT5_PHASE_ERROR {87.180}\
		   CONFIG.CLKOUT6_JITTER {83.768}\
		   CONFIG.CLKOUT6_PHASE_ERROR {87.180}\
		   CONFIG.CLKOUT7_JITTER {83.768}\
		   CONFIG.CLKOUT7_PHASE_ERROR {87.180}] [get_bd_cells clk_wiz_0]

#re-connect the DPU's lowpower signale
disconnect_bd_net /clk_wiz_0_clk_out8 [get_bd_pins DPUCZDX8G_1/ap_clk_2]
disconnect_bd_net /proc_sys_reset_6_peripheral_aresetn [get_bd_pins DPUCZDX8G_1/ap_rst_n_2]
connect_bd_net [get_bd_pins DPUCZDX8G_1/ap_clk_2] [get_bd_pins clk_wiz_0/clk_out6]
connect_bd_net [get_bd_pins DPUCZDX8G_1/dpu_clk_dsp_ce] [get_bd_pins clk_wiz_0/clk_out6_ce]
connect_bd_net [get_bd_pins DPUCZDX8G_2/dpu_clk_dsp_ce] [get_bd_pins clk_wiz_0/clk_out7_ce]
connect_bd_net [get_bd_pins DPUCZDX8G_1/ap_rst_n_2] [get_bd_pins proc_sys_reset_5/peripheral_aresetn]

